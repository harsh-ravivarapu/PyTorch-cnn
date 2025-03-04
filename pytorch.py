import torch # importing PyTorch
from torch import nn # nn in PyTorch is an module that provides with building block of a neural network
from torch.optim import Adam # Adam is an optimiser
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import  ToTensor

#get_data
train = datasets.MNIST(root='data',download=True, train=True, transform=ToTensor()) 
'''
Download the dataset
MNIST is an imgae classification dataset which has 10 classes(0-9)
next we have to specify where we want to download it so ----> root="data" (download it into dat afolder/directory in the root directory)
download=true (here it true because we want to download it)
train=True (its to mention thta we will want to partision the data to train)
next we mention idf we want to transform our data so, as here we want to transform our data into tensors( transform - ToTensor)
'''
#--------x--------x-------x-----x----

'''
here we create a dataset useing Dataloder
then we pass our train partion created earlier and me mention our batch size as 32(training batch, converting it into batches of 32 images)
'''
dataset = DataLoader(train,32, shuffle= True) 

#---------x----------x---------x-------

'''
here we will create outr neural network class
(1,32,(3,3)) in conv2d 
a) 1 -> in_channels (number of input channels, as the images here are gray scale (black and white) it = 1 ;; incase of RGB images it would be = 3
b) 32 -> out_channels (its the number of filters,( kernals you want tarck) (its basically number of unique filters you want to track (think of it like differnt distinct rows on an ML dataset like titanic data set))
c)(3,3) -> is the kernal size or size of the filter, its for scanning the image using an 3x3 matrix(box)
increasing out_channel(b) will in increase computauional costs as more filters mean more convolutions (i.e mathmatical operation that is used to detect edges, tectures, shapes etc) 
increasing kernel size too will increase computational costs 
'''

class ImageClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.model =nn.Sequential(
		nn.Conv2d(1,32,(3,3)),#convalutional neural network 
		nn.ReLU(),
		nn.Conv2d(32,64,(3,3)),#convalutional neural network 
		nn.ReLU(),
		nn.Conv2d(64,64,(3,3)),#convalutional neural network 
		nn.ReLU(),
		nn.Flatten(),
		nn.Linear(64*(28-6)*(28-6), 10)
		)
		
	def forward(self,x):
		return self.model(x)
	

clf = ImageClassifier().to("cpu")	
opt = Adam(clf.parameters(), lr=1e-3) ## Adam is an optimiser, it works by adjustiong the learning rates of each parameter based on history of its gradients (gradient is the measure of change in weights with regard to change in error)

loss_fn = nn.CrossEntropyLoss()
		
#training loop

if __name__ == "__main__":
	for epoch in range(10):
		for batch in dataset:
			x, y = batch
			x, y = x.to("cpu").float ,y.to("cpu")
			yhat = clf(x)
			loss = loss_fn(yhat,y)
			
			
			# apply backpropogation
			
			opt.zero_grad()
			loss.backward()
			opt.step()
			
		print(f"epoch: {epoch}, loss: {loss.item()}")
			
	
		
		




		