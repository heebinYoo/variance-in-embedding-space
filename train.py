
import logging
import dataset
import utils
import losses

import os

import torch
import numpy as np
import matplotlib
#matplotlib.use('agg', warn=False, force=True)
import matplotlib.pyplot as plt
import time
import argparse
import json
import random
from utils import JSONEncoder, json_dumps

from torch.distributions import normal

parser = argparse.ArgumentParser(description='Training ProxyNCA++')
parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='config.json')
parser.add_argument('--embedding-size', default = 512, type=int, dest = 'sz_embedding')
parser.add_argument('--batch-size', default = 32, type=int, dest = 'sz_batch')
parser.add_argument('--epochs', default = 40, type=int, dest = 'nb_epochs')
parser.add_argument('--log-filename', default = 'example')
parser.add_argument('--workers', default = 0, type=int, dest = 'nb_workers')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mode', default='train', choices=['train', 'trainval', 'test'],
                    help='train with train data or train with trainval')
parser.add_argument('--lr_steps', default=[1000], nargs='+', type=int)
parser.add_argument('--source_dir', default='', type=str)
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--eval_nmi', default=True, action='store_true')
parser.add_argument('--recall', default=[1,2,4,8], nargs='+', type=int)
parser.add_argument('--init_eval', default=False, action='store_true')
parser.add_argument('--no_warmup', default=False, action='store_true')
parser.add_argument('--apex', default=False, action='store_true')
parser.add_argument('--warmup_k', default=5, type=int)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # set random seed for all gpus

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('log'):
    os.makedirs('log')

curr_fn = os.path.basename(args.config).split(".")[0]

out_results_fn = "log/%s_%s_%s_%d.json" % (args.dataset, curr_fn, args.mode, args.seed)

config = utils.load_config(curr_fn+'/'+args.dataset+'.json')

dataset_config = utils.load_config('dataset/config.json')


if args.source_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['source'])
    dataset_config['dataset'][args.dataset]['source'] = os.path.join(args.source_dir, bs_name)
if args.root_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['root'])
    dataset_config['dataset'][args.dataset]['root'] = os.path.join(args.root_dir, bs_name)

if args.apex:
    from apex import amp

#set NMI or recall accordingly depending on dataset. note for cub and cars R=1,2,4,8
if (args.mode =='trainval' or args.mode == 'test'):
    if args.dataset == 'sop' or args.dataset == 'sop_h5':
        args.recall = [1, 10, 100, 1000]
    elif 'cub' in args.dataset or 'cars' in args.dataset:
        args.eval_nmi = True

args.nb_classes = config['nb_classes']
args.tr_classes = config['tr_classes']
args.nb_epochs = config['nb_epochs']
args.sz_batch = config['sz_batch']
args.nb_samples = config['nb_samples']
args.sz_embedding = config['sz_embedding']
if 'warmup_k' in config:
    args.warmup_k = config['warmup_k']

transform_key = 'transform_parameters'
if 'transform_key' in config.keys():
    transform_key = config['transform_key']


args.log_filename = '%s_%s_%s_%d' % (args.dataset, curr_fn, args.mode, args.seed)
if args.mode == 'test':
    args.log_filename = args.log_filename.replace('test', 'trainval')

best_epoch = args.nb_epochs

feat = config['model']['type']()
feat.eval()
in_sz = feat(torch.rand(1,3,256,256)).squeeze().size(0)
feat.train()
emb = torch.nn.Linear(in_sz, args.sz_embedding)
model = torch.nn.Sequential(feat, emb)

if not args.apex:
    model = torch.nn.DataParallel(model)
model = model.cuda()

def save_best_checkpoint(model):
    torch.save(model.state_dict(), 'results/' + args.log_filename + '.pt')

def load_best_checkpoint(model):
    model.load_state_dict(torch.load('results/' + args.log_filename + '.pt'))
    model = model.cuda()
    return model



if args.mode == 'trainval':
    train_results_fn = "log/%s_%s_%s_%d.json" % (args.dataset, curr_fn, 'train', args.seed)
    if os.path.exists(train_results_fn):
        with open(train_results_fn, 'r') as f:
            train_results = json.load(f)
        args.lr_steps = train_results['lr_steps']
        best_epoch = train_results['best_epoch']

train_transform = dataset.utils.make_transform(
    **dataset_config[transform_key]
)
print('best_epoch', best_epoch)

results = {}


if ('inshop' not in args.dataset ):
    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            )
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #pin_memory = True
    )
else:
    #inshop trainval mode
    dl_query = torch.utils.data.DataLoader(
        dataset.load_inshop(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            ),
            dset_type = 'query'
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #pin_memory = True
    )
    dl_gallery = torch.utils.data.DataLoader(
        dataset.load_inshop(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            ),
            dset_type = 'gallery'
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #pin_memory = True
    )


logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)



if args.mode == 'train':
    tr_dataset = dataset.load(
        name = args.dataset,
        root = dataset_config['dataset'][args.dataset]['root'],
        source = dataset_config['dataset'][args.dataset]['source'],
        classes = dataset_config['dataset'][args.dataset]['classes']['train'],
        transform = train_transform
    )
elif args.mode == 'trainval' or args.mode == 'test':
    tr_dataset = dataset.load(
        name = args.dataset,
        root = dataset_config['dataset'][args.dataset]['root'],
        source = dataset_config['dataset'][args.dataset]['source'],
        classes = dataset_config['dataset'][args.dataset]['classes']['trainval'],
        transform = train_transform
    )



num_class_per_batch = config['num_class_per_batch']
num_gradcum = config['num_gradcum']
is_random_sampler = config['is_random_sampler']
if is_random_sampler:
    batch_sampler = dataset.utils.RandomBatchSampler(tr_dataset.ys, args.sz_batch, True, num_class_per_batch, num_gradcum)
else:

    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch, int(args.sz_batch / num_class_per_batch))


dl_tr = torch.utils.data.DataLoader(
    tr_dataset,
    batch_sampler = batch_sampler,
    num_workers = args.nb_workers,
    #pin_memory = True
)

print("===")
if args.mode == 'train':
    dl_val = torch.utils.data.DataLoader(
        dataset.load(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['val'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            )
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #drop_last=True
        #pin_memory = True
    )


criterion = config['criterion']['type'](
    nb_classes = dl_tr.dataset.nb_classes(),
    sz_embed = args.sz_embedding,
    **config['criterion']['args']
).cuda()

opt_warmup = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            'lr': 0
        },
        {
            **{'params': list(emb.parameters()
                              )
               },
            **config['opt']['args']['embedding']

        },

        {
            **{'params': criterion.parameters()}
            ,
            **config['opt']['args']['proxynca']

        },

    ],
    **config['opt']['args']['base']
)

opt = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            **config['opt']['args']['backbone']
        },
        {
            **{'params': list(emb.parameters()
                              )
               },
            **config['opt']['args']['embedding']
        },

        {
            **{'params': criterion.parameters()},
            **config['opt']['args']['proxynca']
        },

    ],
    **config['opt']['args']['base']
)

if args.apex:
    [model, criterion], [opt, opt_warmup] = amp.initialize([model, criterion], [opt, opt_warmup], opt_level='O1')
    model = torch.nn.DataParallel(model)

if args.mode == 'test':
    with torch.no_grad():
        logging.info("**Evaluating...(test mode)**")
        model = load_best_checkpoint(model)
        if 'inshop' in args.dataset:
            utils.evaluate_inshop(model, dl_query, dl_gallery)
        else:
            utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)

    exit()

if args.mode == 'train':
    scheduler = config['lr_scheduler']['type'](
        opt, **config['lr_scheduler']['args']
    )
elif args.mode == 'trainval':
    scheduler = config['lr_scheduler2']['type'](
        opt,
        milestones=args.lr_steps,
        gamma=0.1
        #opt, **config['lr_scheduler2']['args']
    )

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()



if args.init_eval:
    logging.info("**Evaluating initial model...**")
    with torch.no_grad():
        if args.mode == 'train':
            c_dl = dl_val
        else:
            c_dl = dl_ev

        utils.evaluate(model, c_dl, args.eval_nmi, args.recall) #dl_val

it = 0

best_val_hmean = 0
best_val_nmi = 0
best_val_epoch = 0
best_val_r1 = 0
best_test_nmi = 0
best_test_r1 = 0
best_test_r2 = 0
best_test_r5 = 0
best_test_r8 = 0
best_tnmi = 0


def batch_lbl_stats(y):
    print(torch.unique(y))
    kk = torch.unique(y)
    kk_c = torch.zeros(kk.size(0))
    for kx in range(kk.size(0)):
        for jx in range(y.size(0)):
            if y[jx] == kk[kx]:
                kk_c[kx] += 1

def get_centers(dl_tr):
    c_centers = torch.zeros(dl_tr.dataset.nb_classes(), args.sz_embedding).cuda()
    n_centers = torch.zeros(dl_tr.dataset.nb_classes()).cuda()
    for ct, (x,y,_) in enumerate(dl_tr):
        with torch.no_grad():
            m = model(x.cuda())
        for ix in range(m.size(0)):
            c_centers[y] += m[ix]
            n_centers[y] += 1
    for ix in range(n_centers.size(0)):
        c_centers[ix] = c_centers[ix] / n_centers[ix]

    return c_centers


def LargestEig(x, center=True, scale=True):
    n, p = x.size()
    ones = torch.ones(n).view([n, 1]).cuda()
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n).cuda() - h
    X_center = torch.mm(H.double(), x.double())
    covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
    scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).cuda().double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
    eigenvalues, eigenvectors = torch.linalg.eigh(scaled_covariance, 'U')
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


prev_lr = opt.param_groups[0]['lr']
lr_steps = []

print(len(dl_tr))

if not args.no_warmup:
    #warm up training for 5 epochs
    logging.info("**warm up for %d epochs.**" % args.warmup_k)
    for e in range(0, args.warmup_k):
        for ct, (x,y,_) in enumerate(dl_tr):
            opt_warmup.zero_grad()
            m = model(x.cuda())
            loss = criterion(m, y.cuda(),low=False)
            if args.apex:
                with amp.scale_loss(loss, opt_warmup) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_warmup.step()
        logging.info('warm up ends in %d epochs' % (args.warmup_k-e))

for e in range(0, args.nb_epochs):
    #if args.mode == 'trainval':
    #    scheduler.step(e)

    if args.mode == 'train':
        curr_lr = opt.param_groups[0]['lr']
        print(prev_lr, curr_lr)
        if curr_lr != prev_lr:
            prev_lr = curr_lr
            lr_steps.append(e)

    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    tnmi = []

    opt.zero_grad()
    for ct, (x, y, _) in enumerate(dl_tr):
        it += 1

        emb = model(x.cuda())
        #############################################################
        #get first eigenvector
        labels_set = list(set(y.cpu().numpy()))
        label_to_indices = {label: np.where(y.cpu().numpy() == label)[0] for label in labels_set}

        eig_vecs=torch.zeros((args.tr_classes,args.sz_embedding)).cuda()
        low_confidence_sample =  torch.zeros((args.sz_batch))
        # 새로운 샘플의 임베딩위치 :
        new_samples_target = torch.full((args.nb_samples,1),labels_set[0]+args.tr_classes,dtype=int)
        for i in range(len(labels_set)):
            num_sample =len(label_to_indices[labels_set[i]])
            emp_center = emb[label_to_indices[labels_set[i]]].mean(0)
            if num_sample >1 :
                eig_vecs[i],scaled_covariance = LargestEig(emb[label_to_indices[labels_set[i]]])

            else : eig_vecs[i] = emb[label_to_indices[labels_set[i]]]

            #샘플의 방향으로 아이겐벡터방향 정해줘야함
            if torch.dot(emp_center,eig_vecs[i]) <0 :
                eig_vecs[i]=eig_vecs[i]*-1
            new_sample_centroid = emp_center + (eig_vecs[i] * torch.norm(emp_center) * 2)

            #샘플의 크기가 emprical mean을 mean으로 갖는 1d gaussian이라고 가정해서 기준이되는 sigma 값에 따라 low confidence sample을 뽑음
            sigma=0.1
            length_std = torch.std(torch.norm(emb[label_to_indices[labels_set[i]]],dim=1))
            inds = torch.where(torch.norm(emb[label_to_indices[labels_set[i]]],dim=1)<torch.norm(emp_center)-sigma*length_std)[0].cpu().detach()
            if len(inds) != 0:
                low_confidence_sample[label_to_indices[labels_set[i]][inds]] = 1

            new_sample_distribution = normal.Normal(new_sample_centroid, torch.norm(emp_center)/8)
            new_samples_emb = new_sample_distribution.sample([args.nb_samples])
            emb = torch.cat((emb,new_samples_emb.float()),0)
            if i != 0:
                new_samples_target = torch.cat((new_samples_target.flatten(),torch.full((args.nb_samples,1),labels_set[i]+args.tr_classes,dtype=int).flatten()),0)

        #loss_aug = criterion(emb[args.sz_batch:],new_samples_target.cuda(),low=True)
        loss_high =0
        loss_low = 0
        if len(torch.where(low_confidence_sample == 0)[0]) != 0:
            loss_high = criterion(emb[torch.where(low_confidence_sample == 0)[0]], y[torch.where(low_confidence_sample == 0)[0]].cuda(), low=False)
        if len(torch.where(low_confidence_sample == 1)[0]) != 0:
            loss_low = criterion(emb[torch.where(low_confidence_sample == 1)[0]], y[torch.where(low_confidence_sample == 1)[0]].cuda(), low=True)
        loss = loss_high + loss_low

        if args.apex:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())

        if (ct + 1) % 1 == 0:
            opt.step()
            opt.zero_grad()



    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch[-20:]))
    print('it: {}'.format(it))
    #print(opt)
    logging.info(
        "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            time_per_epoch_2 - time_per_epoch_1
        )
    )

    model.losses = losses
    model.current_epoch = e

    if e == best_epoch:
        break

    if args.mode == 'train':
        with torch.no_grad():
            logging.info("**Validation...**")
            nmi, recall = utils.evaluate(model, dl_val, args.eval_nmi, args.recall)

        chmean = (2 * nmi * recall[0]) / (nmi + recall[0])

        scheduler.step(chmean)

        if chmean > best_val_hmean:
            best_val_hmean = chmean
            best_val_nmi = nmi
            best_val_r1 = recall[0]
            best_val_r2 = recall[1]
            best_val_r4 = recall[2]
            best_val_r8 = recall[3]
            best_val_epoch = e
            best_tnmi = torch.Tensor(tnmi).mean()

        if e == (args.nb_epochs - 1):
            #saving last epoch
            results['last_NMI'] = nmi
            results['last_hmean'] = chmean
            results['best_epoch'] = best_val_epoch
            results['last_R1'] = recall[0]
            results['last_R2'] = recall[1]
            results['last_R4'] = recall[2]
            results['last_R8'] = recall[3]

            #saving best epoch
            results['best_NMI'] = best_val_nmi
            results['best_hmean'] = best_val_hmean
            results['best_R1'] = best_val_r1
            results['best_R2'] = best_val_r2
            results['best_R4'] = best_val_r4
            results['best_R8'] = best_val_r8

        logging.info('Best val epoch: %s', str(best_val_epoch))
        logging.info('Best val hmean: %s', str(best_val_hmean))
        logging.info('Best val nmi: %s', str(best_val_nmi))
        logging.info('Best val r1: %s', str(best_val_r1))
        logging.info(str(lr_steps))

    if args.mode == 'trainval':
        scheduler.step(e)


if args.mode == 'trainval':
    save_best_checkpoint(model)

    with torch.no_grad():
        logging.info("**Evaluating...**")
        model = load_best_checkpoint(model)
        if 'inshop' in args.dataset:
            best_test_nmi, (best_test_r1, best_test_r10, best_test_r20, best_test_r30, best_test_r40, best_test_r50) = utils.evaluate_inshop(model, dl_query, dl_gallery)
        else:
            best_test_nmi, (best_test_r1, best_test_r2, best_test_r4, best_test_r8) = utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)
        #logging.info('Best test r8: %s', str(best_test_r8))
    if 'inshop' in args.dataset:
        results['NMI'] = best_test_nmi
        results['R1']  = best_test_r1
        results['R10'] = best_test_r10
        results['R20'] = best_test_r20
        results['R30'] = best_test_r30
        results['R40'] = best_test_r40
        results['R50'] = best_test_r50
    else:
        results['NMI'] = best_test_nmi
        results['R1'] = best_test_r1
        results['R2'] = best_test_r2
        results['R4'] = best_test_r4
        results['R8'] = best_test_r8

if args.mode == 'train':
    print('lr_steps', lr_steps)
    results['lr_steps'] = lr_steps

with open(out_results_fn,'w') as outfile:
    json.dump(results, outfile)

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
