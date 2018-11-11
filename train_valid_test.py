# -*- coding: utf-8 -*-

import torch
from utils import timeit
from torch.autograd import Variable

#@timeit
def train_epoch(model, tr_loader, criterion, optimizer, lr, results):
    
    train_loss = 0     
    correct, total = 0, 0       
    
    # Run minibaches from the training dataset
    for i, (X, labels) in enumerate(tr_loader):
        
        X, labels = Variable(X), Variable(labels)
        
        # Forward pass
        model.zero_grad()
        y_pred = model(X)
        s, preds = torch.max(y_pred.data, 1)
        
        # Compute loss 
        loss = criterion(y_pred, labels)            
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect stats 
        train_loss += loss.item()
        model.collect_stats(lr)

        # Compute and store epoch results
        total += y_pred.size(0)
        correct += int(sum(preds == labels)) 
#        if i % 20 == 0: print(correct/total)
        
    lss = round((train_loss / i+1), 3)
    acc = round((correct / total) * 100, 2)
    results.train_accy.append(acc)    
    results.train_loss.append(lss)
    return lss, acc


def valid_epoch(model, ts_loader, criterion, results):
    
    valid_loss = 0
    correct, total = 0, 0
    
    with torch.no_grad():
        for i, (X, labels) in enumerate(ts_loader):
            
            X, labels = Variable(X), Variable(labels)
            
            # Forward pass
            y_pred = model(X)
            s, preds = torch.max(y_pred.data, 1)
            
            # Compute loss 
            loss = criterion(y_pred, labels)           
            valid_loss += loss.item()
            
            # Compute and store epoch results
            total += y_pred.size(0)
            correct += int(sum(preds == labels)) 
#            if i % 20 == 0: print(correct/total)
    
    lss = round((valid_loss/i+1), 3)
    acc = round((correct / total) * 100, 3)
    results.valid_loss.append(lss)
    results.valid_accy.append(acc)
    return lss, acc
