import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import bitalino as BT
import random
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from numpy import mean, median

class bitalino(object):

	def __init__(self, macAddress = '98:D3:31:80:48:08', samplingRate = 1000, channels = [0,1,2,3,4,5]):

		self.board = BT.BITalino(macAddress)

		# Sampling Rate (Hz)
		self.dT = 1.0 / float(samplingRate)
		self.samplingRate = samplingRate

		# Number of samples acquired before transmission is made
		# This is a parameter, it influences the board to PC delay
		# and the power consumption of the board
		self.nSamp = 200

		# Initialize the board
		self.channels = channels

	def initialize_time_series(self, total_time):
		# Declare the time stamp and output lists
		self.cont = 0
		totSamp = int(total_time * self.samplingRate)
		self.y = np.zeros((totSamp, len(self.channels)))
		self.t = np.zeros(totSamp)
		self.classification = np.zeros(totSamp)
		
	def sample(self, sampTime):
		# Initialize variables to sample the correct amount of samples
		# Two loops are needed, every transmission sends self.nSamp samples
		# Thus, the number of transmissions is given by the total samples needed
		# (found using the total time) divided by the samples per transmission
		Sampnum = int(sampTime * self.samplingRate)
		transNum = int(Sampnum / self.nSamp)
		for n in range(transNum):
			samples = self.board.read(self.nSamp)
			#print samples
			for sample in samples:
				self.y[self.cont,:] = sample[5:]
				self.y[self.cont,4] = self.y[self.cont,4] * 16
				self.y[self.cont,5] = self.y[self.cont,5] * 16
				#print self.y
				self.t[self.cont] = self.dT * self.cont
				self.cont = self.cont + 1
		return self.t, self.y

	def plot(self, t, y, plotChan = 'all'):
		if plotChan == 'all':
			plotChan = self.channels
		cont = 0
		for chan in plotChan:
			plt.figure()
			# Prepare the y by transforming it into a list
			ytmp = [i[cont] for i in  y]
			plt.plot(t,ytmp)
			plt.title("EMG Signal - Analog Channel " + str(chan))
			plt.xlabel("Time (s)")
			plt.ylabel("EMG Amplitude (AU)")
			plt.show()
			cont = cont + 1

	def training_interface(self, mov_number, reps_per_mov = 3, resting_time = 2, execution_time = 3):
		''' This function serves as an interactive interface for executing
		the signal acquisition for the neural network. As such, it will ask
		the user to input several parameters and it will execute the training routine
		and it will prepare the data to train the neural network
		
		movs will be a list of dictionaries in which every entry contains the name of the
		movement and the classification (binary, true or false). The classification is just
		a list, the time will be given by a shared time variable, as well as the EMG signal
		'''
		self.movs = []
		# Slightly cryptic: The total duration of the time series will be given by the number of
		# movements time the number of repetitions time the resting time (between movements) + execution time
		self.initialize_time_series(mov_number * reps_per_mov * (resting_time + execution_time))
		for i in range(mov_number):
			st = raw_input("Insert the name of the movement: ")
			self.movs.append({'ID' : i, 'Name' : st, 'Classification': np.zeros(len(self.t))})
		# Start the real training algorithm, the user will be told to do random movements
		mov_counter = np.ones(mov_number) * reps_per_mov
		print "Starting the training algorithm... Relax your muscles"
		self.wait(resting_time)
		self.board.start(self.samplingRate, self.channels)
		for i in range(mov_number * reps_per_mov):
			random.seed()
			mov_type = random.randint(0, mov_number - 1)
			while mov_counter[mov_type] == 0:
				mov_type = random.randint(0, mov_number - 1)
			for i in self.movs:
				if i['ID'] == mov_type:
					# The user prepares to execute the movement, the signal is sampled (it will be classified as no movement)
					print "Prepare ", i['Name'], " movement..."
					self.sample(resting_time)
					# Save the counter to update the class correctly
					tmpcont = self.cont
					print "Execute ", i['Name'], " movement NOW!"
					self.sample(execution_time)
					i['Classification'][tmpcont:self.cont] = 1
					print "Stop!"
			mov_counter[mov_type] = mov_counter[mov_type] - 1
		for i in self.movs:
			self.classification = self.classification + (i['ID'] + 1) * i['Classification']
		return self.t, self.y, self.classification

	def wait(self, dt):
		t0 = time.time()
		for i in range(dt):
			print dt-i, "..."
			while time.time() - t0 < (i+1):
				pass

	def save_training(self):
		pres_time = time.strftime("%Y%m%d-%H%M")
		np.savetxt('data/' + pres_time + '_emg.txt', self.y_proc)
		np.savetxt('data/' + pres_time + '_time.txt', self.t_proc)
		np.savetxt('data/' + pres_time + '_class.txt', self.classification_proc)
		np.savetxt('data/' + pres_time + '_net_out.txt', self.out)

	def load_training(self, timestamp):
		# NB!! Since the saving is very raw and does NOT preserve information about which channels are used,
		# the following will only be accurate if there are six channels
		lt = np.loadtxt('data/' + timestamp + '_time.txt')
		self.initialize_time_series(len(lt) / 1000)
		self.t_proc = lt
		ldemg = np.loadtxt('data/' + timestamp + '_emg.txt')
		self.y_proc = ldemg
		self.classification_proc = np.loadtxt('data/' + timestamp + '_class.txt')

	def init_classifier(self, hidden_units = 20):
		data = ClassificationDataSet(len(self.channels), nb_classes=5)
		# Prepare the dataset
		for i in range(len(self.classification_proc)):
			data.appendLinked(self.y_proc[i], self.classification_proc[i])
		# Make global for test purposes
		self.data = data
		# Prepare training and test data, 75% - 25% proportion
		self.testdata, self.traindata = data.splitWithProportion(0.25)
		#self.traindata._convertToOneOfMany()
		#self.testdata._convertToOneOfMany()
		# CHECK the number of hidden units
		fnn = buildNetwork(self.traindata.indim, hidden_units, self.traindata.outdim)
		# CHECK meaning of the parameters
		trainer = BackpropTrainer(fnn, dataset=self.traindata, momentum=0, verbose=True, weightdecay=0.01)
		return fnn, trainer, data

	def classify(self, net, trainer):
		trainer.trainUntilConvergence(self.traindata,200)
		print "inizio"
		print self.traindata['target']
		print "fine"
		#trnresult = percentError( roundtrainer.testOnClassData(), self.traindata['target'] )
		#tstresult = percentError( trainer.testOnClassData(dataset=self.testdata), self.testdata['target'] )
		print "epoch: %4d" % trainer.totalepochs
		self.out = net.activateOnDataset(self.traindata)
		cnt = 0
		for i in range(len(self.out)):
			print round(self.out[i]), self.traindata['target'][i] 
			if round(self.out[i]) != self.traindata['target'][i]:
				cnt = cnt + 1
		print "Train error = ", (float(cnt) / float(len(self.out))) * 100, " %"
		self.out = net.activateOnDataset(self.testdata)
		cnt = 0
		for i in range(len(self.out)):
			if round(self.out[i]) != self.testdata['target'][i]:
				cnt = cnt + 1
		print "Test error = ", (float(cnt) / float(len(self.out))) * 100, " %"
		# TODO - Thresholding to convert into discrete classes
		self.out = net.activateOnDataset(self.data)
		plt.plot(self.classification_proc,'r')
		plt.hold(True)
		plt.plot(self.out,'b')
		plt.show()
		return trainer, trnresult, tstresult

	def window_rms(self, a, window_size):
  		a2 = np.power(a,2)
  		window = np.ones(window_size)/float(window_size)
  		return np.sqrt(np.convolve(a2, window, 'same'))

	def data_process(self, factor=0.05, rms_width=500):
		for i in range(len(self.channels)):
			tmp = [b[i] for b in self.y]
			tmp = tmp - mean(tmp)
			res = self.window_rms(tmp, rms_width)
			cont = 0
			for j in self.y:
				j[i] = res[cont]
				cont = cont + 1
		# Remove the 0 class (between the different movements)
		self.y = self.y[self.classification != 0][:]
		self.t = self.t[self.classification != 0][:]
		self.t = np.linspace(0,len(self.t)*self.dT,len(self.t))
		self.classification = self.classification[self.classification != 0]
		# Introduce y_proc and classification_proc that will contain the downsampled signal
		self.y_proc = []
		self.classification_proc = []
		self.t_proc = []
		num_samp = self.samplingRate * factor
		num_it = int(len(self.classification) / num_samp)
		# Downsample the signal by factor (average every factor seconds window)
		for i in range(num_it):
			tmp_row = []
			for col in range(len(self.channels)):
				vect=[int(b[col]) for b in self.y[i*num_samp:(i+1)*num_samp]]
				tmp_row.append(mean(vect))
			self.t_proc.append(i*factor)
			self.y_proc.append(tmp_row)
			self.classification_proc.append(self.classification[i*num_samp])
		#self.plot(self.t,self.y)
		plt.figure()
		plt.hold(True)
		plot_colors = ['b','r','g','k','m','c']
		# Plot the superimposed signals for debugging reasons
		for i in range(len(self.channels)):
			plt.plot(self.t, [b[i] for b in self.y],plot_colors[i])
		plt.show()
		return self.t_proc, self.y_proc, self.classification_proc

if __name__ == '__main__':
	'''
	Arguments: sample: choose whether to sample or to just load the data (1 = sample, 0 = load)
	filename: needed if the load mode is selected to tell which file to load
	'''
	bt = bitalino('98:D3:31:80:48:08',1000,[0,1,2,3,4,5])
	if int(sys.argv[1]) == 1:
		# Sample
		# Experiments made with parameters 5,3,3,2
		bt.training_interface(5,3,2,2)
		bt.data_process(factor = 0.05, rms_width = 500)
	elif int(sys.argv[1]) == 0:
		# Load data
		try:
			bt.load_training(str(sys.argv[2]))
		except:
			raise IOError("Input file not found")
	bt.board.close()
	net, trainer, _ = bt.init_classifier()
	trainer = bt.classify(net, trainer)
	bt.save_training()
	print "Done classifying"
	while True:
		bt = bitalino('98:D3:31:80:48:08',1000,[0,1,2,3,4,5])
		print "Testing acquisition"
		bt.training_interface(1,1,2,2)
		bt.board.close()
		bt.data_process()
		_, _, test_data = bt.init_classifier()
		test_out = net.activateOnDataset(test_data)
		print mean(test_out)
		print median(test_out)
		plt.plot(test_out,'m')
		plt.show()