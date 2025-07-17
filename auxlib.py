from pyedflib import highlevel
#import pyedflib as plib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft
import scipy.signal as signal

r1 = [3,7,11]   # Realized One Hand (Left or Right)
i1 = [4,8,12]   # Imagined One Hand (Left or Right)
r2 = [5,9,13]   # Realized Two (Hands or Feet)
i2 = [6,10,14]  # Imagined Two (Hands or Feet)

codes = {'IL':(i1,'T1',0),'IR':(i1,'T2',1),'ILR':(i2,'T1',2),'IF':(i2,'T2',3),\
		'RL':(r1,'T1',4),'RR':(r1,'T2',5),'RLR':(r2,'T1',6), 'RF':(r2,'T2',7)}

def loadEEG(subject, record, path_files=os.path.join(os.getcwd(), "files")):  #lectura del .edf
	# formatting
	if isinstance(record, int):
		record = "{:02d}".format(record)
	if isinstance(subject, int):
		subject = "{:03d}".format(subject)

	path = os.path.join(path_files, "S" + subject, "S" + subject + "R" + record +'.edf')

	signals, signal_headers, header = highlevel.read_edf(path)

	return signals, signal_headers, header


def GetSignal(subject, tarea, sample_rate=160, segment_length=1280, selected_channels=None):
	'''
	GetSignal():
	Entrada:
	- subject: número de sujeto
	- tarea: 'IL', 'IR', 'ILR', 'IF', 'RL', 'RR', 'RLR', 'RF' (ver codes)
	- sample_rate: tasa de muestreo (Hz)
	- segment_length: longitud deseada del segmento (número de muestras)
	- selected_channels: lista de nombres de canales a seleccionar, si es None, usa todos.
	- 640 corresponde a los 4 seg a 160 Hz

	Salida:
	- sujeto: número de sujeto
	- data: lista de arrays de señal, cada array tiene tamaño (n_canales, segment_length)
	- labels: lista de etiquetas correspondientes a cada segmento
	- channels: nombres de canales seleccionados
	'''

	data = []  # Almacena los recortes de señales
	labels = []  # Almacena los numeros correspondientes a cada task
	channels = []  # Nombres de los canales seleccionados

	# Obtener información de la tarea
	run, T, label = codes[tarea]

	per_run = []
	# Recorrer los índices en 'run' (r1, i1, r2, i2)
	for irec in run:
		# Cargar las señales usando el sujeto y registro
		signals, signal_headers, header = loadEEG(subject, irec)

		# Extraer los nombres de los canales
		channel_names = [header['label'] for header in signal_headers]

		# Si no se especifican canales seleccionados, usar todos los canales disponibles
		if selected_channels is None:
			selected_channels = channel_names  # Usa todos los canales disponibles

		# Filtrar solo los canales seleccionados
		try:
			selected_indices = [channel_names.index(ch) for ch in selected_channels]
		except ValueError as e:
			print(f"Error: Canal {e} no encontrado en los nombres de canales disponibles \
			{channel_names}")
			return subject, [], [], []

		# Filtrar las señales según los índices de canales seleccionados
		signals = signals[selected_indices, :]
		channels = [channel_names[i] for i in selected_indices]

		# Buscar anotaciones relevantes (buscando índices)
		indices = []
		index = 0
		for notas in header['annotations']:
			if notas[2] == T:
				indices.append(index)
			index += 1
			
		points = 400.0 # points guarda cuántos puntos antes del estímulo vamos a usar

		for i in indices:
			event_index = header['annotations'][i][0]
			ti = int(sample_rate * event_index - points) 
			tf = int(sample_rate * event_index + 640)  
			recorte = signals[:, ti:tf]

			# Verificar longitud del recorte
			if recorte.shape[1] == segment_length:
				data.append(recorte)  # Agregar recorte a la lista de datos
				labels.append(label)  # Agregar etiqueta correspondiente
			else:
				print(f'Se descarta {subject} - {tarea}: longitud de recorte incorrecta '
				f'({recorte.shape[1]} en lugar de {segment_length})')
		per_run.append(len(indices))
	# Verificar si no se encontraron segmentos válidos
	if len(data) == 0:
		print(f'Se descarta {subject} - {tarea}: no se encontraron segmentos válidos.')
		return subject, [], [], []

	

	return subject, data, per_run, labels, channels

def task_labels(header):	
	t0_labels = []
	t1_labels = []
	t2_labels = []
	for task in range(len(header['annotations'])):
		if header['annotations'][task][2] == 'T1':
			t1_labels.append(header['annotations'][task][0])
		elif header['annotations'][task][2] == 'T2':
			t2_labels.append(header['annotations'][task][0])
		else:
			t0_labels.append(header['annotations'][task][0])
	return t0_labels, t1_labels, t2_labels

def showEEG(signals, ti, tf, time, header, electrodes=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
	t0_labels, t1_labels, t2_labels = task_labels(header)
	fig, axes = plt.subplots(len(electrodes), 1, figsize=(20, len(electrodes)), sharex=True)

	for ax, i in zip(axes, electrodes):
		ax.plot(time[ti:tf], signals[i][ti:tf], color='purple', linewidth=0.8)
		for spine in ['top', 'right', 'left', 'bottom']:
			ax.spines[spine].set_visible(False)

		ax.set_yticks([])
		ax.set_xticks([])

		ax.text(-0.02, 0.5, f'Ch {i+1}', transform=ax.transAxes, fontsize=20, 
				verticalalignment='center', horizontalalignment='right')
		
		for t1 in t1_labels:
			ax.axvline(x=t1, color='red')
		for t2 in t2_labels:
			ax.axvline(x=t2, color='blue')
		for t0 in t0_labels:
			ax.axvline(x=t0, color='black')
	# Show only the last x-axis
	axes[-1].spines['bottom'].set_visible(True)
	axes[-1].set_xlim(time[ti], time[tf-1])
	axes[-1].set_xticks(range(0, int((tf - ti)/160), max(1, (tf - ti) // 4800)))  # Adjust tick marks
	axes[-1].set_xlabel("Time (s)", fontsize=20)

	plt.subplots_adjust(left=0.05)  # Adjust layout to fit labels

	plt.tight_layout()
	plt.show()
	return fig

# PSD
def psd_percentage(x, fs, nfft):


	"""
	Power Density Spectrum Porcentual
	# Parameters
	# x: Signal to convert
	# fs: sample frequency
	# nfft: fft length, padded with 0
	# Outputs
	# Psd_per: Power Spectarl Density in %, the sum of all vector is 1 (is equal to integrate 0 - fs/2) 
	# Pxx: Power Spectral Density
	# X: FFT mod of x
	# f: frequency scale.
	"""

	N = nfft
	X = abs(fft(x, n=nfft))/N
	Pxx = X**2
	X = X[0:int(N/2)+1]
	X[1:int(N/2)] = 2*X[1:int(N/2)] # duplicate except 0

	Pxx = Pxx[0:int(N/2)+1]

	Pxx[1:int(N/2)] = 2*Pxx[1:int(N/2)]

	#Psd = (1/fs*N)*abs(fft(x, n=nfft))**2
	#Psd = Psd[0:int(N/2)+1]
	#Psd[1:int(N/2)] = 2*Psd[1:int(N/2)] # duplicate except 0

	f, S = signal.periodogram(x, fs,  nfft=nfft, scaling='spectrum')

	#psd_sum = sum(Psd)

	#Psd_per= Psd/psd_sum

	#Psd_per= (nfft/fs)*(Psd/psd_sum)

	#f = (fs/N)*np.arange(0, int(N/2)+1, 1) = S

	Psd_per = S
	f_dif = f[1]-f[0]

	psd_sum = sum(Psd_per) * 2 * f_dif # factor 2 for the duplicate except 0
	if psd_sum == 0:
		psd_sum = 1  # avoid division by zero

	Psd_per= Psd_per/psd_sum

	return Psd_per, Pxx, X, f

# FILTERS

def notch_filter(data, f0, Q, fs):

	'''
	# Parameters
	# data: data to be filtered
	# f0: frequency to filter
	# Q: Filter Quality Factor, Range Values 1-100, Typical 30-40
	# fs: Sample frequency
	#
	# Output
	# array filtrated signal
	'''
	# IIR notch filter using signal.iirnotch
	b, a = signal.iirnotch(f0, Q, fs)

	# Compute magnitude response of the designed filter
	freq, h = signal.freqz(b, a, fs=fs)

	return signal.filtfilt(b, a, data)
	

# Higpass Filtering

def butter_highpass_filtering(data_channel, order, f_cutoff, fs):
	
	butter_filter = signal.butter(N=order, Wn=f_cutoff, output='sos', fs=fs, btype='highpass')
	
	high_pass_filtered_eeg = signal.sosfiltfilt(butter_filter, data_channel)

	return high_pass_filtered_eeg

def butter_highpass_filtering_all_channels(data, order, f_cutoff, fs):
	# filtering each channel
	for i in range(0, data.shape[0]):
		data[i] = butter_highpass_filtering(data[i], order=order, f_cutoff=f_cutoff, fs=fs)        
	return data

def denoising_notch(session, task, f0, Q, fs):
	'''
	# Function description:
	# Denoising signal for each channel (1-64) extracting f0 
	# ---------------------------------------------------------------------------------
	# Parameters
	# sessions: Patient / Subject / Animal - Ej: sessions = [50]
	# tasks: array of expermient runs R3, R4, ... R14. 
	# Examples tasks = [3, 7, 11], tasks = [4, 8, 12], tasks = [5, 8, 13] , [6, 10, 14]
	# f0: frequency to filter
	# Q: Filter Quality Factor, Range Values 1-100, Typical 30-40
	# fs: Sample frequency
	#
	# Output
	# array filtrated signal - 64 eeg channels , signal length 125 sec x 160

	# transpose Matrix to save .csv (Revise)
	'''
	
	data, signal_headers, header = loadEEG(subject=session, record=task)
	t0_labels, t1_labels, t2_labels = task_labels(header)
	time = np.linspace(0, len(data[0])/160, len(data[0]))
			
	# filtering noise
	for i in range(0, data.shape[0]):
		data[i] = notch_filter(data[i], f0=60, Q=30, fs=160)
				
		# Save Data
		#data_t = np.transpose(data)
		#pd.DataFrame(data_t, ).to_csv(f'media/S{session:03d}R{run:02d}_denoised_{f0}_Hz.csv')

	return data, t0_labels, t1_labels, t2_labels




# TASK SEGMENTATION

def task_fragmentation(data, channel, t0_labels, t1_labels, t2_labels, t_left, t_right):
	'''
	#Task Segementation / Fragmentations
	# Description: Create segments for each T
	# Parameters:
	# data : data of all channels for specific session/patient, run number (3 = R3, 4 = R4, etc)
	# dimensions: array[channels (64), total experiment samples (20000)]
	# channel: signal channel, dimensions: scalar
	# t0_labels, t1_labels, t2_labels: event labels, dimensions: array[7 or 8 or 15,] (depending repetitions)
	# t_left: time left (time in seconds before event t1, or t2), dimensions: scalar
	# t_right: time right (time in seconds after event t1, or t2), dimensions: scalar

	'''
	data_t0 = []
	data_t1 = []
	data_t2 = []
	
	for i in range(len(t0_labels)):
		t0_index = int(t0_labels[i]*160)
		delta = int(4.1*160)
		t0_segment = data[channel, t0_index: t0_index+delta]
		data_t0.append(t0_segment)
	data_t0 = np.array(data_t0)
	
	
	for j in range(len(t1_labels)):
		t1_index = int(t1_labels[j]*160)
		delta_left = int(t_left*160)
		delta_right = int(t_right*160)
		t1_segment = data[channel, t1_index-delta_left: t1_index+delta_right]
		data_t1.append(t1_segment)
	data_t1 = np.array(data_t1)
	
	for k in range(len(t2_labels)):
		t2_index  = int(t2_labels[k]*160)
		delta_left = int(t_left*160)
		delta_right = int(t_right*160)
		t2_segment = data[channel, t2_index-delta_left: t2_index+delta_right]
		data_t2.append(t2_segment)
	data_t2 = np.array(data_t2)
	
	return data_t0, data_t1, data_t2