from bayesByBackprop import ShuffleNetV2, GaussianVariationalInference
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import sys


def main():
	batch_size = 64
	type_model = "shuffle-net_standard_batchnorm"
	data = datasets.CIFAR10('./data', train=True,
		                transform=transforms.Compose([
		                    transforms.RandomCrop(32, padding=4),
		                    transforms.RandomHorizontalFlip(),
		                    transforms.ToTensor(),
		                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		                    ]))


	train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
	    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	    ])), batch_size=batch_size, shuffle=False)
	resume = False
	num_samples = 5
	net = ShuffleNetV2(net_size=.5)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6, last_epoch=-1)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5,
		                                               verbose=True)
	if os.path.isfile(f"./checkpoint/cifar10/{type_model}.pth.tar"):
	    checkpoint = torch.load(f"./checkpoint/cifar10/{type_model}_last.pth.tar")
	    net.load_state_dict(checkpoint['net'])
	    epoch = checkpoint['epoch']
	    best_score = checkpoint['acc']
	    acc_train = checkpoint['acc_train']
	    acc_test = checkpoint['acc_test']
	    loss_train = checkpoint['loss_train']
	    loss_test = checkpoint['loss_test']
	    optimizer.load_state_dict(checkpoint['optimizer'])
	    # if best_score not in acc_test[-5:]:
	    #     for param in optimizer.param_groups:
	    #         param["lr"] *= .5
	    #         print(param["lr"])
	    #         print("lower lr")
	    scheduler.load_state_dict(checkpoint['scheduler'])
	    print(f"current best score: {best_score:.2f}%")
	    print("loss_train: ", loss_train[-1])
	    print("loss_test: ", loss_test[-1])
	    print("acc_test: ", acc_test[-1])
	    print("acc_train: ", acc_train[-1])

	else:
	    epoch = 0
	    best_score = 0
	    loss_train = list()
	    loss_test = list()
	    acc_train = list()
	    acc_test = list()

	net.samples = num_samples
	vi = GaussianVariationalInference()
	epochs = 200
	print(net.name)
	for i in range(epoch, epochs):
	    net.train()
	    correct = 0
	    total = 0
	    # print(scheduler.get_lr()[0])
	    m = int(len(train_loader.dataset) / batch_size)
	    for batch_idx, (x, y) in enumerate(train_loader, 1):
		beta = 1.1 ** (m - batch_idx) / ((1.1 ** m - 1)*10)
		optimizer.zero_grad()
		out, kl = net(x)
		loss = vi(out, y, kl, beta)
		neg_likelihood = F.cross_entropy(out, y)
		loss_train.append(neg_likelihood.item())

		beta_kl = (kl*beta).item()
		loss.backward()
		optimizer.step()
		_, predicted = torch.max(out.data, 1)
		total += y.size(0)
		correct += predicted.eq(y.data).cpu().sum()
		if not batch_idx % 1:
		    sys.stdout.write('\r')
		    sys.stdout.write(f"Train\tepoch [{i+1}/{epochs}]\t Iter: {batch_idx}/{len(train_loader)}\t"
		                     f"loss: {loss.item():.3e}\t"
		                     f"Acc@1: {(100*correct.float()/total):.2f}%\t"
		                     f"neg_likelihood: {neg_likelihood.item():.2f}\t"
		                     f"kl_beta: {beta_kl:.3e}")
		    sys.stdout.flush()
	    acc_train.append((100*correct.float()/total).item())
	    net.eval()
	    total = 0
	    correct = 0
	    print('\n')
	    with torch.no_grad():
		for batch_idx, (x, y) in enumerate(test_loader, 1):
		    out, _ = net(x)
		    total += y.size(0)
		    _, predicted = torch.max(out.data, 1)
		    correct += predicted.eq(y.data).cpu().sum()
		    neg_likelihood = F.cross_entropy(out, y).item()
		    loss_test.append(neg_likelihood)
		    if not batch_idx % 1:
		        sys.stdout.write('\r')
		        sys.stdout.write(f"Test\tepoch [{i + 1}/{epochs}]\t Iter: {batch_idx}/{len(test_loader)}\t"
		                         f"Acc@1: {(100 * correct.float() / total):.2f}%\t"
		                         f"neg_likelihood: {neg_likelihood:.2f}\t")
		        sys.stdout.flush()
	    acc = 100 * correct.float() / total
	    print(f"\n| Validation Epoch #{i+1}\t\t\t Acc@1: {acc:.2f}%\t"
		  f"mean loss: {loss_test[-1]:.2f}")
	    acc_test.append(acc.item())
	    scheduler.step(acc)
	    state = {
		'net': net.state_dict(),
		'acc': max(acc_test),
		'epoch': i,
		'loss_test': loss_test,
		'loss_train': loss_train,
		'acc_train': acc_train,
		'acc_test': acc_test,
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler.state_dict()
	    }
	    if acc > best_score:
		print(f"| Saving Best model...\t\t\tTop1 = {acc:.2f}%")
		if not os.path.isdir('checkpoint'):
		    os.mkdir('checkpoint')
		save_point = './checkpoint/' + "cifar10" + os.sep
		if not os.path.isdir(save_point):
		    os.mkdir(save_point)
		torch.save(state, save_point + type_model + '.pth.tar')
		best_score = acc
	    if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')
	    save_point = './checkpoint/' + "cifar10" + os.sep
	    if not os.path.isdir(save_point):
		os.mkdir(save_point)
	    torch.save(state, save_point + type_model + '_last.pth.tar')
	    state = dict()
