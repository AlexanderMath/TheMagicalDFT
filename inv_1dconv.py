import numpy as np
from numpy.fft import fft2, ifft2, ifftn, fftn
import matplotlib.pyplot as plt
from scipy import signal 

# create data. 
A = np.random.normal(0, 1, size=(10,))
k = np.random.normal(0, 1, size=(3,))

# compute convolutions. (?? No flip ??)
conv 	= signal.convolve(		A, k, mode="same")
fftconv = signal.fftconvolve(	A, k, mode="same")
fft		= ifftn( fftn( A, (2*10-1,) ) * fftn( k, (2*10-1,)), (2*10-1, ) ).real[1: 11]

lst 	= [conv, fftconv, fft]
names	= ["conv", "fftconv", "fft"]


# compare convolutions (should be the same)
for a in lst: print(a.shape)

fig, ax = plt.subplots(4, 3)

for j in range(3): 
	ax[0, j].imshow(lst[j].reshape(1, 10))
	ax[0, j].set_title(names[j])

	ax[0,j].set_xticks([])
	ax[0,j].set_yticks([])
	ax[0,j].set_xticklabels([])
	ax[0,j].set_yticklabels([])


for i in range(3):
	for j in range(3): 
		print(i, j, np.allclose(lst[i], lst[j]))
		ax[i+1,j].imshow(np.abs(lst[i] - lst[j]).reshape(1, 10)) # be sure to each check size are not relative. 

		ax[i+1,j].set_xticks([])
		ax[i+1,j].set_yticks([])
		ax[i+1,j].set_xticklabels([])
		ax[i+1,j].set_yticklabels([])
		ax[i+1,j].set_title("|%s - %s|"%(names[i], names[j]))


plt.tight_layout()

# compute convolution and invert it (naively). 
# Naive in the sense we pad k as much as A, maybe this can be done with less padding. 
A_part 	= fftn( A, (2*10-1,)) 
k_part 	= fftn( k, (2*10-1,))
inside 	= A_part * k_part
conv	= ifftn( inside, (2*10-1,) )
 
rec 	= fftn( conv, (2*10-1, ) )
assert np.allclose(inside, rec)

rec		= rec * 1 / k_part
assert np.allclose(A_part, rec)

rec		= ifftn( rec, (2*10-1, ))
assert np.allclose( rec[:10].real, A )

fig, ax = plt.subplots(4, 1)

ax[0].imshow(A.reshape(1, 10).real)
ax[0].set_title("A")
ax[1].imshow(conv.reshape(1, 19).real)
ax[1].set_title("conv (all)")
ax[2].imshow(rec.reshape(1, 19).real)
ax[2].set_title("rec (all)")
ax[3].imshow(rec[0: 10].reshape(1, 10).real)
ax[3].set_title("rec")

for i in range(4): 
	ax[i].set_xticks([])
	ax[i].set_yticks([])
	ax[i].set_xticklabels([])
	ax[i].set_yticklabels([])

plt.pause(10**6)


