import os
import sys
import random

import PIL
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import trange

import numpy as np
import cv2
import timm
import torch
import torchvision
import torchvision.transforms.functional as TF
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch.nn.functional as F
import tensorflow as tf
import pandas as pd
import albumentations as A
from tqdm.autonotebook import tqdm

def to_pil(img):
  return PIL.Image.fromarray(img, 'RGB')

def gen_random(dim):
  return torch.from_numpy(np.random.RandomState(0).randn(1, dim)).cuda()

def perpre(img):
  return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

class Stylegan2Gen(torch.nn.Module):
  def __init__(self, model_path, stylegan_dir, truncation_psi=0.5):
    super().__init__()
    sys.path.insert(1, stylegan_dir)
    import legacy
    with open(model_path, 'rb') as f:
      self.G = legacy.load_network_pkl(f)['G_ema'].cuda().eval()
    self.truncation = truncation_psi

  @property
  def z_dim(self):
    return self.G.z_dim

  def forward(self, z, truncation=None):
    label = torch.zeros([1, self.G.c_dim]).cuda()
    if truncation is None:
      truncation = self.truncation
    return self.G(z, label, truncation_psi=truncation)

  def gen_pil(self, z, truncation=None):
    with torch.no_grad():
      img = perpre(self.forward(z, truncation))
      return to_pil(img[0].cpu().numpy())

class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ImageEncoder(torch.nn.Module):
    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class CLIPModel(torch.nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=2048,
        text_embedding=768,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class Pars(torch.nn.Module):
    def __init__(self, dims, batch_size=16):
        super(Pars, self).__init__()
        self.z = torch.nn.Parameter(torch.zeros(batch_size, dims).normal_(std=1).cuda())

    def forward(self):
      return self.z

# Load the model

def plot(model, loss, iter, odir, lats):
  best = torch.topk(loss, k=1, largest=False)[1].item()
  if best > args.batch_size-1:
    best = args.batch_size-1
  lts = lats()[best]
  torch.save(lats()[best], os.path.join(odir, "final.pt"))
  model.gen_pil(lts.unsqueeze(0)).save(os.path.join(odir, f"{iter}.png"))
  return lts

def ascend_txt(model, perceptor, t, nom, lats, la, lb):
  out = model(lats())
  cutn, sideX, sideY = out.size()[1:]
  p_s = []
  for ch in range(cutn):
    size = int(sideX*torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
    offsetx = torch.randint(0, sideX - size, ())
    offsety = torch.randint(0, sideY - size, ())
    apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
    apper = torch.nn.functional.interpolate(apper, (224,224), mode='nearest')
    p_s.append(apper)
  into = torch.cat(p_s, 0)
  into = nom((into + 1) / 2)

  image_features = perceptor.image_encoder(into)
  image_embeddings = perceptor.image_projection(image_features)
  iii = image_embeddings

  llls = lats()
  lat_l = torch.abs(1 - torch.std(llls, dim=1)).mean() + torch.abs(torch.mean(llls)).mean() + 4*torch.clamp_max(torch.square(llls).mean(), 1)

  for array in llls:
    mean = torch.mean(array)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    lat_l = lat_l + torch.abs(kurtoses) / llls.shape[0] + torch.abs(skews) / llls.shape[0]
  return la*lat_l, -lb*torch.cosine_similarity(t, iii, dim=-1)

def train(i, odir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb):
  optimizer.zero_grad()
  a, b = ascend_txt(model, perceptor, t, nom, lats, la, lb)
  loss = a + b.mean()
  loss.backward()
  optimizer.step()

  if i % plot_every == 0:
    plot(model, b, i, odir, lats)

def final(odir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb):
  with torch.no_grad():
    np.save(os.path.join(odir, 'final'), plot(model, ascend_txt(model, perceptor, t, nom, lats, la, lb)[1], 'final', odir, lats).cpu().numpy())

def imagine(text, model_path, lr=.07, seed=0, num_epochs=200, total_plots=20, batch_size=16, outdir=None, stylegan2_dir="stylegan2-ada-pytorch", clip_dir="CLIP", la=1, lb=100, truncation_psi=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptor = CLIPModel().to(device)
    modelCustom = torch.load('/content/drive/MyDrive/clip-stylised-maps.pt')
    perceptor.load_state_dict(modelCustom)
    model = Stylegan2Gen(model_path, stylegan2_dir, truncation_psi)
    im_shape = perpre(model(gen_random(model.z_dim)))[0].size()
    sideX, sideY, channels = im_shape

    torch.manual_seed(seed)
    lats = Pars(model.z_dim, batch_size).cuda()
    optimizer = torch.optim.Adam(lats.parameters(), lr)

    nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    tx = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_query = tx([text])
    batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = perceptor.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = perceptor.text_projection(text_features)

    ta = F.normalize(text_embeddings, p=2, dim=-1)
    t = ta.cuda().detach().clone()
    outdir = (text if outdir is None else outdir)
    if not os.path.isdir(outdir):
      os.mkdir(outdir)
    plot_every = int(num_epochs/total_plots)
    for i in trange(num_epochs):
        train(i, outdir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb)
    final(outdir, plot_every, model, perceptor, optimizer, t, nom, lats, la, lb)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='stylegan2ada-image')
    parser.add_argument('-i', '--text', default="Your cat looks like the devil!")
    parser.add_argument('-n', '--network', required=True)
    parser.add_argument('-e', '--num-epochs', default=200, type=int)
    parser.add_argument('-p', '--total-plots', default=20, type=int)
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--truncation_psi', default=0.5, type=float)
    parser.add_argument('--lr', default=0.07, type=float)
    parser.add_argument('--la', default=1, type=float, help='Loss-factor a')
    parser.add_argument('--lb', default=100, type=float, help='Loss-factor b')
    parser.add_argument('-s', '--stylegan2-dir', default="stylegan2-ada-pytorch")
    parser.add_argument('-c', '--clip-dir', default="CLIP")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()
    imagine(args.text, args.network, args.lr, args.seed, args.num_epochs, args.total_plots, args.batch_size, args.outdir, args.stylegan2_dir, args.clip_dir, args.la, args.lb, args.truncation_psi)
