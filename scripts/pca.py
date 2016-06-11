from music21 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

bachBundle = corpus.search('bwv')
bachBundle = bachBundle.search('4/4')

index =0
data = {}
for n in range(len(bachBundle)):
	data[n] = {}
	for i in range(30,100):
		data[n][i] = 0

for n in range(len(bachBundle)):

	myPiece = bachBundle[n].parse()

	for m in myPiece.flat.getElementsByClass('Note'):
		data[n][m.midi] +=1

	print 'Number %i' % n

new_data = np.array([data[0].values()]).astype(np.float64)
new_data /= np.sum(new_data)

for index in range(len(bachBundle)):
	temp = np.array([data[index].values()]).astype(np.float64)
	temp /= np.sum(temp)

	new_data =  np.concatenate((new_data,  temp)  , axis=0)

print 'Statistics gathered!'



save = new_data


###############################################################################

bachBundle = corpus
bachBundle = bachBundle.search('4/4')

index =0
data = {}
for n in range(700, 2500):
	data[n] = {}
	for i in range(30,100):
		data[n][i] = 0

for n in range(700, 2500):

	myPiece = bachBundle[n].parse()

	for m in myPiece.flat.getElementsByClass('Note'):
			data[n][m.midi] +=1

	print 'Number %i' % n

new_data = np.array([data[700].values()])
new_data /= np.sum(new_data)

for index in range(700, 2500):

	temp = np.array([data[index].values()]).astype(np.float64)
	temp /= np.sum(temp)

	new_data =  np.concatenate( (new_data,  temp )  , axis=0)

print 'Statistics gathered!'



X = new_data


pca = PCA(n_components=2)

d = np.concatenate((save,X))

X_r = pca.fit(d).transform(d)


x_1 = [ i[0] for i in X_r]
x_2 = [ i[1] for i in X_r]

y_1 = x_1[:len(save)]
y_2 = x_2[:len(save)]

x_1 = x_1[len(save):]
x_2 = x_2[len(save):]

plt.figure()
plt.plot(y_1, y_2, '.b')
plt.plot(x_1, x_2, '.g')

for i in [str(x) for x in range(1,9) ]:

	myPiece = converter.parse('out-'+i+'.xml')

	new_data ={}
	save = [0,0,0,0]

	for i in range(30,100):
			new_data[i] = 0



	for m in myPiece.flat.getElementsByClass('Note'):
			new_data[m.midi] +=1


	new_data = np.array([new_data.values()]).astype(np.float64)
	new_data /= np.sum(new_data)

	N = pca.transform(new_data)

	plt.plot(N[0][0], N[0][1], 'oc')

plt.title('PCA')
plt.legend(['Bach', 'other' , 'generated' ], loc=2)
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()
