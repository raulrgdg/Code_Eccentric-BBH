import numpy as np
import matplotlib.pyplot as plt

#Loading the arrays computed before
best_q = np.load()
best_mchirp = np.load()
best_overlap = np.load()

num_elements=len(best_mchirp)

#Signal population
s_e=np.linspace(0.01, 0.3, num_elements) #Adjusting for each case
s_q=np.linspace(0.1,1.0, num_elements )

#2 color-plot for fitting factor
plt.figure(figsize=(8, 6))
cax = plt.imshow(best_overlap, extent=(s_q.min(), s_q.max(), s_e.min(), s_e.max()), origin='lower', aspect='auto', cmap='viridis')
cbar = plt.colorbar(cax)
cbar.set_label('Maximum Overlap',  labelpad=15, fontsize=15)
cbar.ax.tick_params(labelsize=12)
plt.xlabel('q', fontsize=15)
plt.ylabel('e', fontsize=15, labelpad=19)
plt.title(f'Fitting factors', fontsize=15)
plt.savefig('fig1')
plt.show()


#2 color-plot for bias in chirp mass
bias_mchirp=np.log(best_mchirp/6.08364)
fig = plt.figure(figsize=(8, 6))
cax = plt.imshow(bias_mchirp, extent=(s_q.min(), s_q.max(), s_e.min(), s_e.max()), origin='lower', aspect='auto', cmap='Spectral',vmin=-0.1,vmax=0.1)
cbar = fig.colorbar(cax)
cbar.set_label(r'$ln(\frac{\mathcal{M}^*}{\mathcal{M}})$',labelpad=15, fontsize=18)
cbar.ax.tick_params(labelsize=12)
plt.xlabel('q', fontsize=15,)
plt.ylabel('e', fontsize=15, labelpad=19)
plt.title('Chirp mass study', fontsize=15)
plt.savefig('fig2')

#2 color-plot for bias in mass ratio
s_q=np.linspace(0.1,1.0, num_elements )
bias_q=np.zeros((num_elements,num_elements))
for i in range(num_elements):
    for j in range(num_elements):
        bias_q[i,j]=np.log(best_q[i,j]/s_q[j])

fig = plt.figure(figsize=(8, 6))
cax = plt.imshow(bias_q, extent=(s_q.min(), s_q.max(), s_e.min(), s_e.max()), origin='lower', aspect='auto', cmap='Spectral',vmin=-2.5,vmax=2.5)
cbar = fig.colorbar(cax)
cbar.set_label(r'$ln(\frac{q^*}{q})$', labelpad=15, fontsize=18)
cbar.ax.tick_params(labelsize=12)
plt.xlabel('q', fontsize=15)
plt.ylabel('e', fontsize=15, labelpad=19)
plt.title('Mass ratio study', fontsize=14)
plt.savefig('fig3')
























'''
load_directory = '/home/raulrgdg/gravi/optimal_parameters'

s_e=np.linspace(0.01, 0.9, 40)

best_q = np.load(os.path.join(load_directory, 'best_q_1.npy'))
best_mchirp = np.load(os.path.join(load_directory, 'best_mchirp_1.npy'))
best_overlap = np.load(os.path.join(load_directory, 'best_overlap_1.npy'))



plt.plot(best_q, s_e, 'r*', label='best_q')
plt.plot(best_mchirp, s_e , 'b*', label='best_mchirp')
plt.plot(best_overlap, s_e , 'y*', label='best_overlap')

# Añadir leyenda y títulos
plt.legend()
plt.title('Gráfico de parámetros óptimos')
plt.xlabel('s_e')
plt.ylabel('Valores')


plt.savefig('parameters_vs_s_e')
'''
