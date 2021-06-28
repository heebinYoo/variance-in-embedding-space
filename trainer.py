import torch
import numpy as np

from torch.distributions import normal

import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils

cuda = torch.cuda.is_available()
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']


def recall(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None, binary=False):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

    sim_matrix = torch.mm(feature_vectors, gallery_vectors.t().contiguous())
    if binary:
        sim_matrix = sim_matrix / feature_vectors.size(-1)

    if gallery_labels is None:
        sim_matrix.fill_diagonal_(0)
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = sim_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list


def fit(train_loader,train_loader_unpick, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics,epoch,batch_size=150, n_classes=5)
        for metric in metrics :
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
            if metric.name() == "Accuracy":
                train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
                plot_weight = model.get_weight().cpu()
                plot_embeddings(train_embeddings_baseline, train_labels_baseline, epoch,plot_weight,n_classes=5)
                #unpick으로 recall@k NMI 계산 해야함


                train_embeddings_unpick, train_labels_unpick = extract_embeddings(train_loader_unpick, model)
                NMI,recall_k,feats_coll = utils.eval_metrics_one_dataset(train_labels_unpick,train_embeddings_unpick,cuda,[1,2,3,4])
                #recall_k=recall(train_embeddings_unpick, train_labels_unpick, rank=[1,2,3,4])
                message += '\t{}: {}'.format('NMI', NMI)
                message += '\t{}: {}'.format('recall_k', recall_k)
                plot_embeddings(train_embeddings_unpick, train_labels_unpick, epoch+0.1,[],n_classes=5)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        #recall 함수 구현해야함,,,



        #val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        #val_loss /= len(val_loader)

        #message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,val_loss)
        #for metric in metrics:
        #    message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def LargestEig(x, center=True, scale=True):
    n, p = x.size()
    ones = torch.ones(n).view([n, 1]).cuda()
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n).cuda() - h
    X_center = torch.mm(H.double(), x.double())
    covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
    scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).cuda().double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
    eigenvalues, eigenvectors = torch.symeig(scaled_covariance, True)
    """
    total = eigenvalues.sum()
    if k>=1:
        index = 511-k
        components = (eigenvectors[:, index:])
    else :
        eigsum = 0
        index = 0
        for i in range(512):
            eigsum = eigsum + eigenvalues[511-i]
            if eigsum >= total*k:
                index = 511-i
                break;
        components = (eigenvectors[:, index:])
    """

    return eigenvectors[:,1] ,scaled_covariance


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics,epoch,batch_size,n_classes):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs, emb = model(*data,[])


        labels_set = list(set(target.cpu().numpy()))
        label_to_indices = {label: np.where(target.cpu().numpy() == label)[0] for label in labels_set}
        eig_vecs=torch.zeros((n_classes,2)).cuda()
        low_confidence_sample =  torch.zeros((batch_size))
        # 새로운 샘플의 임베딩위치 :
        new_samples_target = torch.full((n_classes,1),5,dtype=int)
        for i in range(len(labels_set)):
            num_sample =len(label_to_indices[labels_set[i]])
            emp_center = emb[label_to_indices[labels_set[i]]].mean(0)
            if num_sample >1 :
                eig_vecs[i],scaled_covariance = LargestEig(emb[label_to_indices[labels_set[i]]])

            else : eig_vecs[i] = emb[label_to_indices[labels_set[i]]]
            #center의 위치에 따라 방향을 다르게 해줘야함
            #샘플의 방향으로 아이겐벡터방향 정해줘야함
            if torch.dot(emp_center,eig_vecs[i]) <0 :
                eig_vecs[i]=eig_vecs[i]*-1
            new_sample_centroid = emp_center + (eig_vecs[i] * torch.norm(emp_center) * 2)
            #원점과의 거리가 클래스센트로이드보다 가까운 애들만 체크
            inds = torch.where(torch.norm(emb[label_to_indices[labels_set[i]]],dim=1)<torch.norm(emp_center))[0].cpu().detach()
            low_confidence_sample[label_to_indices[i][inds]]=1

            new_sample_distribution = normal.Normal(new_sample_centroid, torch.norm(emp_center)/8)
            new_samples_emb = new_sample_distribution.sample([n_classes])
            emb = torch.cat((emb,new_samples_emb.float()),0)
            if i != 0:
                new_samples_target = torch.cat((new_samples_target.flatten(),torch.full((n_classes,1),i+n_classes,dtype=int).flatten()),0)
            #새로운샘플들 다 추가한 임베딩 스페이스 점들을 classfication last layer에 넣어


        target = torch.cat((target,new_samples_target.cuda()),0)


        sftm_high = outputs[torch.where(low_confidence_sample == 0)[0]]
        sftm_low= outputs[torch.where(low_confidence_sample == 1)[0]]
        sftm_aug= model([],emb[batch_size:])
        if len(sftm_low)+len(sftm_high)!=batch_size:
            print("이상해요") # batch_size 확인
        eps=1e-7
        exp_log_sftm = torch.exp(sftm_high[:,:n_classes])
        norm_exp_log_sftm = F.normalize(exp_log_sftm.view(exp_log_sftm.size(0), -1), dim=-1, p=1).view(exp_log_sftm.size())
        loss_inputs_high = torch.log(norm_exp_log_sftm+eps)

        loss_inputs_low = sftm_low
        loss_inputs_aug = sftm_aug
        if type(loss_inputs_high) not in (tuple, list):
            loss_inputs_high = (loss_inputs_high,)
        if type(loss_inputs_low) not in (tuple, list):
            loss_inputs_low = (loss_inputs_low,)
        if type(loss_inputs_aug) not in (tuple, list):
            loss_inputs_aug = (loss_inputs_aug,)


        if target is not None:
            target_high = (target[torch.where(low_confidence_sample == 0)[0]],)
            target_low = (target[torch.where(low_confidence_sample == 1)[0]],)
            loss_inputs_high += target_high
            loss_inputs_low += target_low

        if target is not None:
            target_aug = (target[batch_size:],)
            loss_inputs_aug += target_aug

        loss_outputs_high = loss_fn(*loss_inputs_high)
        loss_outputs_low = loss_fn(*loss_inputs_low)
        loss_outputs_aug = loss_fn(*loss_inputs_aug)
        if type(loss_outputs_high) in (tuple, list) :
            loss = loss_outputs_high[0] + 5 * loss_outputs_low[0] + loss_outputs_aug[0]
        else :
            loss = loss_outputs_high + 5 *loss_outputs_low + loss_outputs_aug
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()


        for metric in metrics:
            metric((sftm_high,), (target_high), loss_outputs_high)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

#,xlim=(-300,500), ylim=(-300,500)
def plot_embeddings(embeddings, targets,epoch, eig,n_classes,xlim=[],ylim=[]):
    plt.figure(figsize=(10,10))
    for i in range(2*n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
        if i <n_classes and eig != [] :
            #plt.scatter(eig[i,0]*50, eig[i,1]*50, alpha=1,marker='x', color=colors[i])
            plt.annotate('',(eig[i,0]*100, eig[i,1]*100),(0,0), arrowprops=dict(facecolor=colors[i],width=1,headwidth=1,headlength=1))
            plt.text(eig[i,0]*100, eig[i,1]*100,'{}'.format(i))

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(mnist_classes)
    plt.axvline(x=0, color = 'r') # draw x =0 axes
    plt.axhline(y=0, color = 'r') # draw y =0 axes
    plt.savefig('./images/{}training.png'.format(epoch))
    if epoch==19:
        print("cc")

    plt.clf()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
