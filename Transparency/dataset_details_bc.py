from Transparency.Trainers.DatasetBC import *
#dataset = SST_dataset()
dataset = IMDB_dataset()
#dataset = SST_dataset()
#dataset = SST_dataset()

print('DATASET:',dataset.name,'\n')
# Train data
print('Train data sizes:')
print('total:',len(dataset.train_data.y))
print('pos  :',np.sum(dataset.train_data.y),np.sum(dataset.train_data.y)/len(dataset.train_data.y))
print('neg  :',len(dataset.train_data.y)-np.sum(dataset.train_data.y),1-np.sum(dataset.train_data.y)/len(dataset.train_data.y))
# Dev data
print('Dev data sizes:')
print('total:',len(dataset.dev_data.y))
print('pos  :',np.sum(dataset.dev_data.y),np.sum(dataset.dev_data.y)/len(dataset.dev_data.y))
print('neg  :',len(dataset.dev_data.y)-np.sum(dataset.dev_data.y),1-np.sum(dataset.dev_data.y)/len(dataset.dev_data.y))
# Test data
print('Test data sizes:')
print('total:',len(dataset.test_data.y))
print('pos  :',np.sum(dataset.test_data.y),np.sum(dataset.test_data.y)/len(dataset.test_data.y))
print('neg  :',len(dataset.test_data.y)-np.sum(dataset.test_data.y),1-np.sum(dataset.test_data.y)/len(dataset.test_data.y))

print('Average size of doc:',np.mean([len(i) for i in dataset.train_data.X]))

v=set()
for sen in dataset.train_data.X:
    for w in sen:
        v.add(w)
print('Size of vocab train data:',len(v))

