from Transparency.Trainers.DatasetQA import *
#dataset = get_Babi_1()
#dataset = get_SNLI()
dataset = get_CNN()


print('DATASET:',dataset.name,'\n')
# Train data
print('Train data size:',len(dataset.train_data.P))
print('dev data size:',len(dataset.dev_data.P))
print('test data size:',len(dataset.test_data.P))


print('Average len of doc:',np.mean([len(i) for i in dataset.train_data.P]))
print('Average len of question:',np.mean([len(i) for i in dataset.train_data.Q]))

v_d=set()
for sen in dataset.train_data.P:
    for w in sen:
        v_d.add(w)
print('Size of vocab train data, doc:',len(v_d))
v_q=set()
for sen in dataset.train_data.Q:
    for w in sen:
        v_q.add(w)
print('Size of vocab train data, questions:',len(v_q))

ans_tot=set()
for a in dataset.train_data.A:
    ans_tot.add(a)
print('Total number of answer categories:',len(ans_tot))

print('Average number of answer categories per question:',np.mean([np.sum(i) for i in dataset.train_data.E]))



