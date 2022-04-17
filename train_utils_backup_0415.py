import logging
import os
from models.CNN import CNN
import torch
import torch.nn as nn
import data_loader
import models
from torch import optim
from tqdm import tqdm
import itertools
import utils
import torch.nn.functional as F
from collections import defaultdict

class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
    
    def init_data(self):
        '''
        Initialize the datasets.
        '''
        args = self.args
        self.device = torch.device("cpu")
        self.device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(self.device_count))
        
        self.datasets = {}
        if args.processing_type != 'R_NA':
            raise Exception("processing type not implemented")
        else:
            for source in args.source_name:
                data_root = os.path.join(args.data_dir, source)
                try:
                    Dataset = getattr(data_loader, source)
                except:
                    raise Exception("data name type not implemented")
                self.datasets[source] = Dataset(data_root, args.normlizetype).data_preprare(is_src=True)
                logging.info('source set {} length {}.'.format(source, len(self.datasets[source])))
            
            data_root = os.path.join(args.data_dir, args.target_name)
            try:
                Dataset = getattr(data_loader, args.target_name)
            except:
                raise Exception("data name type not implemented")
            self.datasets['train'], self.datasets['val'] = Dataset(data_root, args.normlizetype).data_preprare()
            logging.info('training set length {}, validation set length {}.'.format(len(self.datasets['train']),
                                                                                    len(self.datasets['val'])))
                         
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(False if x == 'val' else True),
                                                           num_workers=args.num_workers, drop_last=True,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                                                           for x in (['train', 'val'] + args.source_name)}
        self.iters = {x: iter(self.dataloaders[x]) for x in (['train', 'val'] + args.source_name)}
    
    def train_CNN(self):
        self.init_data()
        args = self.args
        self.lr_scheduler = None
        
        self.model = CNN(1, 3).to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        
        best_acc = 0.0
        best_epoch = 0

        for epoch in range((args.max_epoch+1)):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                num_iter = len(self.iters[phase])
                for i in tqdm(range(num_iter), ascii=True):
                    data, labels = utils.get_next_batch(self.dataloaders,
                    						     self.iters, phase,
                    						     self.device)
                    
                    if phase == 'train':
                        # Do the learning process, in val, we do not care about the gradient for relaxing
                        with torch.set_grad_enabled(phase == 'train'):
                            # forward
                            self.optimizer.zero_grad()
                            pred = self.model(data)
                            loss = F.cross_entropy(pred, labels)
                                
                            pred = pred.argmax(dim=1)
                            correct = torch.eq(pred, labels).float().sum().item()
                            
                            epoch_loss += loss
                            epoch_acc += correct/pred.shape[0]
                            
                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred = self.model(data)
                            pred = pred.argmax(dim=1)
                            
                            correct = torch.eq(pred, labels).float().sum().item()
                            epoch_acc += correct/pred.shape[0]

                # Print the train and val information via each epoch
                epoch_acc = epoch_acc/num_iter
                if phase == 'train':
                    logging.info('{}-Loss: {:.4f} {}-Acc: {:.4f}'.format(
                        phase, epoch_loss/num_iter, phase, epoch_acc))
                else:
                    logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))
                
                    # log the best model according to the val accuracy
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
        
    def setup(self):
        """
        Initialize the model, loss and optimizer.
        """
        self.init_data()
        args = self.args
        # Define the model
        try:
            model = getattr(models, args.model_name)
        except:
            raise Exception("model type not implemented")
        self.model = model(in_channel=1, out_channel=args.classes)
        
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implemented")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implemented")

        # Load the checkpoint
        self.start_epoch = 1

        # Invert the model
        self.model.to(self.device)
    
    def train_1src(self):
        args = self.args
        best_acc = 0.0
        best_epoch = 0
        src = args.source_name[0]

        for epoch in range(self.start_epoch, (args.max_epoch+1)):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                num_iter = len(self.iters[phase])
                for i in tqdm(range(num_iter), ascii=True):
                    if phase == 'train':
                        source_data, source_labels = utils.get_next_batch(self.dataloaders,
                        						     self.iters, src,
                        						     self.device)    
                    target_data, target_labels = utils.get_next_batch(self.dataloaders,
                    							 self.iters, phase,
                    							 self.device)
                    
                    if phase == 'train':
                        # Do the learning process, in val, we do not care about the gradient for relaxing
                        with torch.set_grad_enabled(phase == 'train'):
                            # forward
                            self.optimizer.zero_grad()
                            cls_loss, mmd_loss, pred = self.model(source_data, data_tgt = target_data,
                                                                  label_src = source_labels)
                            loss = cls_loss + args.alpha * (mmd_loss)
                                
                            pred = pred.argmax(dim=1)
                            correct = torch.eq(pred, target_labels).float().sum().item()
                            
                            epoch_loss += loss
                            epoch_acc += correct/pred.shape[0]
                            
                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred = self.model(target_data)
                            pred = pred.argmax(dim=1)
                            
                            correct = torch.eq(pred, target_labels).float().sum().item()
                            epoch_acc += correct/pred.shape[0]

                # Print the train and val information via each epoch
                epoch_acc = epoch_acc/num_iter
                if phase == 'train':
                    logging.info('{}-Loss: {:.4f} {}-Acc: {:.4f}'.format(
                        phase, epoch_loss/num_iter, phase, epoch_acc))
                else:
                    logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))
                
                    # log the best model according to the val accuracy
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
                    
                    if args.save_step != 0:
                        if epoch % args.save_step == 0 or epoch == args.max_epoch:
                            # save the checkpoint for other learning
                            model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                            torch.save(model_state_dic,
                                        os.path.join(self.save_dir, '{}-{:.4f}-model.pth'.format(epoch, epoch_acc)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
    def train_2src(self):
        args = self.args
        best_acc = 0.0
        best_epoch = 0
        
        source1_iter = self.iters[args.source_name[0]]
        source2_iter = self.iters[args.source_name[1]]
        
        for epoch in range(self.start_epoch, (args.max_epoch+1)):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                epoch_acc = 0
                epoch_acc1 = 0
                epoch_acc2 = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                num_iter = len(self.iters[phase])
                for i in tqdm(range(num_iter), ascii=True):
                    if phase == 'train':
                        try:
                            source_data, source_labels = source1_iter.next()
                            source_state = 1
                        except Exception:
                            try:
                                source_data, source_labels = source2_iter.next()
                                source_state = 2
                            except Exception:
                                source1_iter = iter(self.dataloaders[args.source_name[0]])
                                source2_iter = iter(self.dataloaders[args.source_name[1]])
                                source_data, source_labels = source1_iter.next()
                                source_state = 1
                        source_data = source_data.to(self.device)
                        source_labels = source_labels.to(self.device)
                        
                    target_data, target_labels = utils.get_next_batch(self.dataloaders,
									 self.iters, phase,
									 self.device)
                    
                    if phase == 'train':
                        # Do the learning process, in val, we do not care about the gradient for relaxing
                        with torch.set_grad_enabled(phase == 'train'):
                            # forward
                            self.optimizer.zero_grad()
                            cls_loss, mmd_loss, l1_loss, pred = self.model(source_data,
                                                                           data_tgt = target_data,
                                                                           label_src = source_labels,
                                                                           mark = source_state)
                            loss = cls_loss + args.alpha * (mmd_loss + l1_loss)
                            
                            pred = pred.argmax(dim=1)
                            correct = torch.eq(pred, target_labels).float().sum().item()
                            
                            epoch_loss += loss
                            epoch_acc += correct/pred.shape[0]
                            
                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred1, pred2 = self.model(target_data)
                            
                            pred = (F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1))/2
                            pred = pred.argmax(dim=1)
                            pred1 = pred1.argmax(dim=1)
                            pred2 = pred2.argmax(dim=1)
                            
                            correct = torch.eq(pred, target_labels).float().sum().item()
                            epoch_acc += correct/pred.shape[0]
                            correct1 = torch.eq(pred1, target_labels).float().sum().item()
                            epoch_acc1 += correct1/pred1.shape[0]
                            correct2 = torch.eq(pred2, target_labels).float().sum().item()
                            epoch_acc2 += correct2/pred2.shape[0]

                # Print the train and val information via each epoch
                if phase == 'train':
                    logging.info('{}-Loss: {:.4f} {}-Acc: {:.4f}'.format(
                        phase, epoch_loss/num_iter, phase, epoch_acc/num_iter))
                else:
                    logging.info('{}-Acc_src1: {:.4f} Acc_src2: {:.4f}'.format(
                        phase, epoch_acc1/num_iter, epoch_acc2/num_iter))
                    epoch_acc = epoch_acc/num_iter
                    
                    # log the best model according to the val accuracy
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
                    
                    if args.save_step != 0:
                        if epoch % args.save_step == 0 or epoch == args.max_epoch:
                            # save the checkpoint for other learning
                            model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                            torch.save(model_state_dic,
                                        os.path.join(self.save_dir, '{}-{:.4f}-model.pth'.format(epoch, epoch_acc)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


    def train_man_moe(self):
        self.init_data()
        args = self.args
        args.all_src = args.source_name + ['train']
        self.device = torch.device("cuda")
        self.start_epoch = 1
        
        F_s = models.FeatureExtractor(input_size=1, num_layers=2, hidden_size=64, dropout=0)
        
        F_p = nn.Sequential(
            models.FeatureExtractor(input_size=1, num_layers=2, hidden_size=64, dropout=0),
            models.MixtureOfExperts(num_layers=2, input_size=64, num_experts=len(args.source_name),
                             hidden_size=128, output_size=64, dropout=0))
        
        C = models.SpAttnMixtureOfExperts(num_layers=2, shared_input_size=64,
                    private_input_size=64, num_experts=len(args.source_name),
                    hidden_size=64, output_size=args.classes, dropout=0)
        
        D = models.CNNDiscriminator(input_size=64, output_size=128,
                    num_src=len(args.all_src), signal_length=1024, dropout=0)
        
        F_s, F_p, C, D = F_s.to(self.device), F_p.to(self.device), \
                         C.to(self.device), D.to(self.device)
                         
        # optimizers
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(*map(list,
                        [F_s.parameters(), C.parameters(), F_p.parameters()]))),
                        lr=args.lr, weight_decay=args.weight_decay)
        optimizerD = optim.Adam(D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # training
        best_avg_acc = 0.0
        num_iter = int(utils.gmean([len(self.dataloaders[x]) for x in args.all_src])/4)
        
        for epoch in range(self.start_epoch, (args.max_epoch+1)):
            F_s.train()
            F_p.train()
            C.train()
            D.train()
            
            # training accuracy
            correct, total = defaultdict(int), defaultdict(int)
            gate_correct = defaultdict(int)
            c_gate_correct = defaultdict(int)
            d_correct, d_total = 0, 0
            
            for i in tqdm(range(num_iter), ascii=True):
                
                # D iterations
                utils.freeze_net(F_s)
                utils.freeze_net(F_p)
                utils.freeze_net(C)
                utils.unfreeze_net(D)
                
                for _ in range(args.n_critic):
                    D.zero_grad()
                    
                    # train on both labeled and unlabeled langs
                    for src in args.all_src:
                        # targets not used
                        d_inputs, _ = utils.get_next_batch(self.dataloaders,
                                                           self.iters, src,
                                                           self.device)
                        idx = args.all_src.index(src)
                        
                        shared_feat = F_s(d_inputs)
                        d_outputs = D(shared_feat)
                        
                        # if token-level D, we can reuse the gate label generator
                        d_targets = utils.get_gate_label(d_outputs, idx, self.device)
                        
                        # D accuracy
                        _, pred = torch.max(d_outputs, -1)
                        d_correct += (pred == d_targets).sum().item()
                        d_total += pred.shape[0]
                        l_d = F.nll_loss(d_outputs, d_targets.view(-1), ignore_index=-1)
                        l_d.backward()
                    optimizerD.step()

                # F&C iteration
                utils.unfreeze_net(F_s)
                utils.unfreeze_net(F_p)
                utils.unfreeze_net(C)
                utils.freeze_net(D)
                F_s.zero_grad()
                F_p.zero_grad()
                C.zero_grad()

                for src in args.source_name:
                    source_inputs, targets = utils.get_next_batch(self.dataloaders,
                                                           self.iters, src,
                                                           self.device)
                    target_inputs, _ = utils.get_next_batch(self.dataloaders,
                                                           self.iters, 'train',
                                                           self.device)
                    idx = args.all_src.index(src)
                    
                    shared_feat = F_s(source_inputs)
                    target_sfeat = F_s(target_inputs)
                    private_feat, target_pfeat, gate_outputs = F_p((source_inputs, target_inputs))
                    c_outputs, c_gate_outputs, l_c = C((shared_feat, private_feat, target_sfeat, target_pfeat))
                    # private_feat, gate_outputs = F_p(source_inputs)
                    # c_outputs, c_gate_outputs = C((shared_feat, private_feat))
                    
                    gate_targets = utils.get_gate_label(gate_outputs, idx, self.device)
                    l_gate = F.cross_entropy(gate_outputs, gate_targets.view(-1), ignore_index=-1)
                    l_c *= 0.00001
                    l_c += args.gate_lw * l_gate
                    _, gate_pred = torch.max(gate_outputs, -1)
                    gate_correct[src] += (gate_pred == gate_targets).sum().item()
                    
                    c_gate_targets = utils.get_gate_label(c_gate_outputs, idx, self.device)
                    l_c_gate = F.cross_entropy(c_gate_outputs, c_gate_targets)
                    l_c += args.gate_lw * l_c_gate
                    _, c_gate_pred = torch.max(c_gate_outputs, -1)
                    c_gate_correct[src] += (c_gate_pred == c_gate_targets).sum().item()
                   
                    l_c += F.nll_loss(c_outputs, targets)
                    _, pred = torch.max(c_outputs, -1)
                    correct[src] += (pred == targets).sum().item()
                    total[src] += pred.shape[0]
                    l_c.backward()

                # update F with D gradients on all langs
                for src in args.all_src:
                    inputs, _ = utils.get_next_batch(self.dataloaders,
                                                     self.iters, src,
                                                     self.device)
                    idx = args.all_src.index(src)
                    shared_feat = F_s(inputs)
                    d_outputs = D(shared_feat)
                    
                    # if token-level D, we can reuse the gate label generator
                    d_targets = utils.get_gate_label(d_outputs, idx, self.device)
                    l_d = F.nll_loss(d_outputs, d_targets.view(-1), ignore_index=-1)
                    if args.lambd > 0:
                        l_d *= -args.lambd
                    l_d.backward()

                optimizer.step()

            # end of epoch
            logging.info('Ending epoch {}'.format(epoch))
            logging.info('D Training Accuracy: {:.3f}%'.format(100.0*d_correct/d_total))
            logging.info('Training accuracy:')
            logging.info('\t'.join(args.source_name))
            logging.info('\t'.join(['%.03f' % (100.0*correct[d]/total[d]) for d in args.source_name]))

            logging.info('Gate accuracy:')
            logging.info('\t'.join(['%.03f' % (100.0*gate_correct[d]/total[d]) for d in args.source_name]))

            logging.info('Tagger Gate accuracy:')
            logging.info('\t'.join(['%.03f' % (100.0*c_gate_correct[d]/total[d]) for d in args.source_name]))
            
            logging.info('Evaluating validation sets:')
            acc = self.evaluate_acc(F_s, F_p, C)
            logging.info('Average validation accuracy: {:.3f}'.format(100.0*acc))

            if acc > best_avg_acc:
                logging.info('New best average validation accuracy: {:.3f}'.format(100.0*acc))
                best_avg_acc = acc
                # with open(os.path.join(opt.model_save_file, 'options.pkl'), 'wb') as ouf:
                #     pickle.dump(opt, ouf)
                # if F_s:
                #     torch.save(F_s.state_dict(),
                #             '{}/netF_s.pth'.format(opt.model_save_file))
                # torch.save(emb.state_dict(),
                #         '{}/net_emb.pth'.format(opt.model_save_file))
                # if F_p:
                #     torch.save(F_p.state_dict(),
                #             '{}/net_F_p.pth'.format(opt.model_save_file))
                # torch.save(C.state_dict(),
                #         '{}/netC.pth'.format(opt.model_save_file))
                # if D:
                #     torch.save(D.state_dict(),
                #             '{}/netD.pth'.format(opt.model_save_file))

        # end of training
        logging.info('Best average validation accuracy: {:.3f}'.format(100.0*best_avg_acc))
    
    def evaluate_acc(self, F_s, F_p, C):
        F_s.eval()
        F_p.eval()
        C.eval()
        iters = iter(self.dataloaders['val'])
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(iters, ascii=True):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                shared_features = F_s(inputs)
                private_feat, _ = F_p(inputs)
                outputs, _ = C((shared_features, private_feat))

                _, pred = torch.max(outputs, -1)
                correct += (pred == targets).sum().item()
                total += pred.shape[0]
            acc = correct / total
            logging.info('Accuracy on {} samples: {:.3f}%'.format(total, 100.0*acc))
        return acc
