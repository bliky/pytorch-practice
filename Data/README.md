# 数据

``` python
# 加载数据
# 定义数据转换 标准化载入数据
transform = transforms.Compose([
                    transforms.ToTensor()
                  ])
# 加载训练数据
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 查看数据
images, labels = next(iter(trainloader))
print('images size: ', images.size())
print('labels size: ', labels.size())

plt.imshow(images.numpy().squeeze()[0])
```