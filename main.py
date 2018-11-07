from flask import (
	Flask, render_template, request, url_for, redirect, jsonify
)

import operator
import re
import string
import numpy as np
from sh import find
import math
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword = stopword_factory.create_stop_word_remover()

app = Flask(__name__)

def remove_punctuation(s):
	"""See: http://stackoverflow.com/a/266162
	"""
	exclude = set(string.punctuation)
	return ''.join(ch for ch in s if ch not in exclude)

def tokenize(text):
	text = remove_punctuation(text)
	text = text.lower()
	return re.split("\W+", text)

def stopword_removal(words):
	sr = {}
	for word in words:
		sr[word] = stopword.remove(word)
	return sr

def stemming(words):
	# s = {}
	# for word in words:
	#     s[word] = stemmer.stem(word)
	return stemmer.stem(words)

def count_words(words):
	wc = {}
	for word in words:
		wc[word] = wc.get(word, 0.0) + 1.0
	return wc

@app.route('/', methods=['GET'])
def main():
	return render_template('index.html')

@app.route('/preprocessing', methods=['GET'])
def preprocessing():
	return render_template('preprocessing.html')

@app.route('/klasifikasi_custom', methods=['GET'])
def klasifikasi_custom():
	return render_template('klasifikasi_custom.html')

@app.route('/klasifikasi_quick', methods=['GET'])
def klasifikasi_quick():
	return render_template('klasifikasi_quick.html')

@app.route('/klasifikasi_input', methods=['GET'])
def klasifikasi_input():
	return render_template('klasifikasi_input.html')

@app.route('/train', methods=['GET'])
def train():
	vocab = {}
	words = {}
	stem = {}
	stopword_removals = {}
	word_counts = {
		"animasi_2d": {},
		"animasi_3d": {},
		"desain_multimedia": {},
		"basis_data": {},
		"pemodelan_perangkat_lunak": {},
		"pemrograman_dasar": {},
		"komputer_terapan_jaringan": {},
		"komunikasi_data": {},
		"sistem_operasi_jaringan": {}
	}
	priors = {
		"animasi_2d": 0.,
		"animasi_3d": 0.,
		"desain_multimedia": 0.,
		"basis_data": 0.,
		"pemodelan_perangkat_lunak": 0.,
		"pemrograman_dasar": 0.,
		"komputer_terapan_jaringan": 0.,
		"komunikasi_data": 0.,
		"sistem_operasi_jaringan": 0.
	}
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
		# ok time to start counting stuff...
		priors[category] += 1
		text = open(f).read()
		# stem = stemming(text)
		words = tokenize(text)
		stopword_removals = stopword_removal(words)
		counts = count_words(stopword_removals)
		for word, count in list(counts.items()):
			# if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
			if word not in vocab:
				vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
			if word not in word_counts[category]:
				word_counts[category][word] = 0.0
			vocab[word] += count
			word_counts[category][word] += count

	result = {
		"tokenize": words,
		"stopword": stopword_removals,
		"word_counts": word_counts
	}

	return render_template('preprocessing.html', result=result)

@app.route('/quick_test', methods=['GET'])
def quick_test():
	vocab = {}
	word_counts = {
		"animasi_2d": {},
		"animasi_3d": {},
		"desain_multimedia": {},
		"basis_data": {},
		"pemodelan_perangkat_lunak": {},
		"pemrograman_dasar": {},
		"komputer_terapan_jaringan": {},
		"komunikasi_data": {},
		"sistem_operasi_jaringan": {}
	}
	priors = {
		"animasi_2d": 0.,
		"animasi_3d": 0.,
		"desain_multimedia": 0.,
		"basis_data": 0.,
		"pemodelan_perangkat_lunak": 0.,
		"pemrograman_dasar": 0.,
		"komputer_terapan_jaringan": 0.,
		"komunikasi_data": 0.,
		"sistem_operasi_jaringan": 0.
	}
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
		# ok time to start counting stuff...
		priors[category] += 1
		text = open(f).read()
		words = tokenize(text)
		stopword_removals = stopword_removal(words)
		# stem = stemming(stopword_removals)
		counts = count_words(stopword_removals)
		for word, count in list(counts.items()):
			# if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
			if word not in vocab:
				vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
			if word not in word_counts[category]:
				word_counts[category][word] = 0.0
			vocab[word] += count
			word_counts[category][word] += count

	response = []
	for f in find("examples"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"

		result = {}
		result['target'] = category
		result['file'] = f
		new_doc = open(f).read()
		words = tokenize(new_doc)
		stop_removals = stopword_removal(words)
		counts = count_words(stop_removals)
		import math

		prior_animasi_2d = (priors["animasi_2d"] / sum(priors.values()))
		prior_animasi_3d = (priors["animasi_3d"] / sum(priors.values()))
		prior_desain_multimedia = (priors["desain_multimedia"] / sum(priors.values()))
		prior_basis_data = (priors["basis_data"] / sum(priors.values()))
		prior_pemodelan_perangkat_lunak = (priors["pemodelan_perangkat_lunak"] / sum(priors.values()))
		prior_pemrograman_dasar = (priors["pemrograman_dasar"] / sum(priors.values()))
		prior_komputer_terapan_jaringan = (priors["komputer_terapan_jaringan"] / sum(priors.values()))
		prior_komunikasi_data = (priors["komunikasi_data"] / sum(priors.values()))
		prior_sistem_operasi_jaringan = (priors["sistem_operasi_jaringan"] / sum(priors.values()))

		log_prob_animasi_2d = 0.0
		log_prob_animasi_3d = 0.0
		log_prob_desain_multimedia = 0.0
		log_prob_basis_data = 0.0
		log_prob_pemodelan_perangkat_lunak = 0.0
		log_prob_pemrograman_dasar = 0.0
		log_prob_komputer_terapan_jaringan = 0.0
		log_prob_komunikasi_data = 0.0
		log_prob_sistem_operasi_jaringan = 0.0

		for w, cnt in list(counts.items()):
			# skip words that we haven't seen before, or words less than 3 letters long
			if w not in vocab or len(w) <= 3:
				continue

			p_word = vocab[w] / sum(vocab.values())
			p_w_given_animasi_2d = word_counts["animasi_2d"].get(w, 0.0) / sum(word_counts["animasi_2d"].values())
			p_w_given_animasi_3d = word_counts["animasi_3d"].get(w, 0.0) / sum(word_counts["animasi_3d"].values())
			p_w_given_desain_multimedia = word_counts["desain_multimedia"].get(w, 0.0) / sum(word_counts["desain_multimedia"].values())
			p_w_given_basis_data = word_counts["basis_data"].get(w, 0.0) / sum(word_counts["basis_data"].values())
			p_w_given_pemodelan_perangkat_lunak = word_counts["pemodelan_perangkat_lunak"].get(w, 0.0) / sum(word_counts["pemodelan_perangkat_lunak"].values())
			p_w_given_pemrograman_dasar = word_counts["pemrograman_dasar"].get(w, 0.0) / sum(word_counts["pemrograman_dasar"].values())
			p_w_given_komputer_terapan_jaringan = word_counts["komputer_terapan_jaringan"].get(w, 0.0) / sum(word_counts["komputer_terapan_jaringan"].values())
			p_w_given_komunikasi_data = word_counts["komunikasi_data"].get(w, 0.0) / sum(word_counts["komunikasi_data"].values())
			p_w_given_sistem_operasi_jaringan = word_counts["sistem_operasi_jaringan"].get(w, 0.0) / sum(word_counts["sistem_operasi_jaringan"].values())
			
			if p_w_given_animasi_2d > 0:
				log_prob_animasi_2d += math.log(cnt * p_w_given_animasi_2d / p_word)
			if p_w_given_animasi_3d > 0:
				log_prob_animasi_3d += math.log(cnt * p_w_given_animasi_3d / p_word)
			if p_w_given_desain_multimedia > 0:
				log_prob_desain_multimedia += math.log(cnt * p_w_given_desain_multimedia / p_word)
			if p_w_given_basis_data > 0:
				log_prob_basis_data += math.log(cnt * p_w_given_basis_data / p_word)
			if p_w_given_pemodelan_perangkat_lunak > 0:
				log_prob_pemodelan_perangkat_lunak += math.log(cnt * p_w_given_pemodelan_perangkat_lunak / p_word)
			if p_w_given_pemrograman_dasar > 0:
				log_prob_pemrograman_dasar += math.log(cnt * p_w_given_pemrograman_dasar / p_word)
			if p_w_given_komputer_terapan_jaringan > 0:
				log_prob_komputer_terapan_jaringan += math.log(cnt * p_w_given_komputer_terapan_jaringan / p_word)
			if p_w_given_komunikasi_data > 0:
				log_prob_komunikasi_data += math.log(cnt * p_w_given_komunikasi_data / p_word)
			if p_w_given_sistem_operasi_jaringan > 0:
				log_prob_sistem_operasi_jaringan += math.log(cnt * p_w_given_sistem_operasi_jaringan / p_word)

		rate = { 
			"animasi_2d": (log_prob_animasi_2d + math.log(prior_animasi_2d)),
			"animasi_2d": (log_prob_animasi_3d + math.log(prior_animasi_3d)),
			"desain_multimedia": (log_prob_desain_multimedia + math.log(prior_desain_multimedia)),
			"basis_data": (log_prob_basis_data + math.log(prior_basis_data)),
			"pemodelan_perangkat_lunak": (log_prob_pemodelan_perangkat_lunak + math.log(prior_pemodelan_perangkat_lunak)),
			"pemrograman_dasar": (log_prob_pemrograman_dasar + math.log(prior_pemrograman_dasar)),
			"komputer_terapan_jaringan": (log_prob_komputer_terapan_jaringan + math.log(prior_komputer_terapan_jaringan)),
			"komunikasi_data": (log_prob_komunikasi_data + math.log(prior_komunikasi_data)),
			"sistem_operasi_jaringan": (log_prob_sistem_operasi_jaringan + math.log(prior_sistem_operasi_jaringan))
		}

		predict = max(rate.iteritems(), key=operator.itemgetter(1))[0]
		result['predict'] = predict
		if(result['predict'] == result['target']):
			result['value'] = 1
		else:
			result['value'] = 0

		response.append(result)

	true_class = 0
	false_class = 0
	for res in response:
		if res['value'] == 0:
			false_class += 1
		else:
			true_class += 1

	data = {
		"akurasi": (float(true_class) / (true_class + false_class)) * 100,
		"datas": response
	}

	return render_template('klasifikasi_quick.html', result=data)

@app.route('/quick_input', methods=['POST'])
def quick_input():
	vocab = {}
	word_counts = {
		"animasi_2d": {},
		"animasi_3d": {},
		"desain_multimedia": {},
		"basis_data": {},
		"pemodelan_perangkat_lunak": {},
		"pemrograman_dasar": {},
		"komputer_terapan_jaringan": {},
		"komunikasi_data": {},
		"sistem_operasi_jaringan": {}
	}
	priors = {
		"animasi_2d": 0.,
		"animasi_3d": 0.,
		"desain_multimedia": 0.,
		"basis_data": 0.,
		"pemodelan_perangkat_lunak": 0.,
		"pemrograman_dasar": 0.,
		"komputer_terapan_jaringan": 0.,
		"komunikasi_data": 0.,
		"sistem_operasi_jaringan": 0.
	}
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
		# ok time to start counting stuff...
		priors[category] += 1
		text = open(f).read()
		words = tokenize(text)
		stopword_removals = stopword_removal(words)
		# stem = stemming(stopword_removals)
		counts = count_words(stopword_removals)
		for word, count in list(counts.items()):
			# if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
			if word not in vocab:
				vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
			if word not in word_counts[category]:
				word_counts[category][word] = 0.0
			vocab[word] += count
			word_counts[category][word] += count

	new_doc = request.form.get('text')
	words = tokenize(new_doc)
	stop_removals = stopword_removal(words)
	counts = count_words(stop_removals)

	prior_animasi_2d = (priors["animasi_2d"] / sum(priors.values()))
	prior_animasi_3d = (priors["animasi_3d"] / sum(priors.values()))
	prior_desain_multimedia = (priors["desain_multimedia"] / sum(priors.values()))
	prior_basis_data = (priors["basis_data"] / sum(priors.values()))
	prior_pemodelan_perangkat_lunak = (priors["pemodelan_perangkat_lunak"] / sum(priors.values()))
	prior_pemrograman_dasar = (priors["pemrograman_dasar"] / sum(priors.values()))
	prior_komputer_terapan_jaringan = (priors["komputer_terapan_jaringan"] / sum(priors.values()))
	prior_komunikasi_data = (priors["komunikasi_data"] / sum(priors.values()))
	prior_sistem_operasi_jaringan = (priors["sistem_operasi_jaringan"] / sum(priors.values()))

	log_prob_animasi_2d = 0.0
	log_prob_animasi_3d = 0.0
	log_prob_desain_multimedia = 0.0
	log_prob_basis_data = 0.0
	log_prob_pemodelan_perangkat_lunak = 0.0
	log_prob_pemrograman_dasar = 0.0
	log_prob_komputer_terapan_jaringan = 0.0
	log_prob_komunikasi_data = 0.0
	log_prob_sistem_operasi_jaringan = 0.0

	for w, cnt in list(counts.items()):
		# skip words that we haven't seen before, or words less than 3 letters long
		if w not in vocab or len(w) <= 3:
			continue

		p_word = vocab[w] / sum(vocab.values())
		p_w_given_animasi_2d = word_counts["animasi_2d"].get(w, 0.0) / sum(word_counts["animasi_2d"].values())
		p_w_given_animasi_3d = word_counts["animasi_3d"].get(w, 0.0) / sum(word_counts["animasi_3d"].values())
		p_w_given_desain_multimedia = word_counts["desain_multimedia"].get(w, 0.0) / sum(word_counts["desain_multimedia"].values())
		p_w_given_basis_data = word_counts["basis_data"].get(w, 0.0) / sum(word_counts["basis_data"].values())
		p_w_given_pemodelan_perangkat_lunak = word_counts["pemodelan_perangkat_lunak"].get(w, 0.0) / sum(word_counts["pemodelan_perangkat_lunak"].values())
		p_w_given_pemrograman_dasar = word_counts["pemrograman_dasar"].get(w, 0.0) / sum(word_counts["pemrograman_dasar"].values())
		p_w_given_komputer_terapan_jaringan = word_counts["komputer_terapan_jaringan"].get(w, 0.0) / sum(word_counts["komputer_terapan_jaringan"].values())
		p_w_given_komunikasi_data = word_counts["komunikasi_data"].get(w, 0.0) / sum(word_counts["komunikasi_data"].values())
		p_w_given_sistem_operasi_jaringan = word_counts["sistem_operasi_jaringan"].get(w, 0.0) / sum(word_counts["sistem_operasi_jaringan"].values())

		if p_w_given_animasi_2d > 0:
			log_prob_animasi_2d += math.log(cnt * p_w_given_animasi_2d / p_word)
		if p_w_given_animasi_3d > 0:
			log_prob_animasi_3d += math.log(cnt * p_w_given_animasi_3d / p_word)
		if p_w_given_desain_multimedia > 0:
			log_prob_desain_multimedia += math.log(cnt * p_w_given_desain_multimedia / p_word)
		if p_w_given_basis_data > 0:
			log_prob_basis_data += math.log(cnt * p_w_given_basis_data / p_word)
		if p_w_given_pemodelan_perangkat_lunak > 0:
			log_prob_pemodelan_perangkat_lunak += math.log(cnt * p_w_given_pemodelan_perangkat_lunak / p_word)
		if p_w_given_pemrograman_dasar > 0:
			log_prob_pemrograman_dasar += math.log(cnt * p_w_given_pemrograman_dasar / p_word)
		if p_w_given_komputer_terapan_jaringan > 0:
			log_prob_komputer_terapan_jaringan += math.log(cnt * p_w_given_komputer_terapan_jaringan / p_word)
		if p_w_given_komunikasi_data > 0:
			log_prob_komunikasi_data += math.log(cnt * p_w_given_komunikasi_data / p_word)
		if p_w_given_sistem_operasi_jaringan > 0:
			log_prob_sistem_operasi_jaringan += math.log(cnt * p_w_given_sistem_operasi_jaringan / p_word)

	rate = { 
		"animasi_2d": (log_prob_animasi_2d + math.log(prior_animasi_2d)),
		"animasi_2d": (log_prob_animasi_3d + math.log(prior_animasi_3d)),
		"desain_multimedia": (log_prob_desain_multimedia + math.log(prior_desain_multimedia)),
		"basis_data": (log_prob_basis_data + math.log(prior_basis_data)),
		"pemodelan_perangkat_lunak": (log_prob_pemodelan_perangkat_lunak + math.log(prior_pemodelan_perangkat_lunak)),
		"pemrograman_dasar": (log_prob_pemrograman_dasar + math.log(prior_pemrograman_dasar)),
		"komputer_terapan_jaringan": (log_prob_komputer_terapan_jaringan + math.log(prior_komputer_terapan_jaringan)),
		"komunikasi_data": (log_prob_komunikasi_data + math.log(prior_komunikasi_data)),
		"sistem_operasi_jaringan": (log_prob_sistem_operasi_jaringan + math.log(prior_sistem_operasi_jaringan))
	}

	predict = max(rate.iteritems(),key=operator.itemgetter(1))[0]
	
	data = {
		"akurasi": predict
	}

	return render_template('klasifikasi_input.html', result=data)

@app.route('/test_fourexamples', methods=['GET'])
def test_fourexamples():
	vocab = {}
	word_counts = {
		"animasi_2d": {},
		"animasi_3d": {},
		"desain_multimedia": {},
		"basis_data": {},
		"pemodelan_perangkat_lunak": {},
		"pemrograman_dasar": {},
		"komputer_terapan_jaringan": {},
		"komunikasi_data": {},
		"sistem_operasi_jaringan": {}
	}
	priors = {
		"animasi_2d": 0.,
		"animasi_3d": 0.,
		"desain_multimedia": 0.,
		"basis_data": 0.,
		"pemodelan_perangkat_lunak": 0.,
		"pemrograman_dasar": 0.,
		"komputer_terapan_jaringan": 0.,
		"komunikasi_data": 0.,
		"sistem_operasi_jaringan": 0.
	}
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
		# ok time to start counting stuff...
		priors[category] += 1
		text = open(f).read()
		words = tokenize(text)
		stopword_removals = stopword_removal(words)
		# stem = stemming(stopword_removals)
		counts = count_words(stopword_removals)
		for word, count in list(counts.items()):
			# if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
			if word not in vocab:
				vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
			if word not in word_counts[category]:
				word_counts[category][word] = 0.0
			vocab[word] += count
			word_counts[category][word] += count

	response = []
	for f in find("examples-4"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"

		result = {}
		result['target'] = category
		result['file'] = f
		new_doc = open(f).read()
		words = tokenize(new_doc)
		stop_removals = stopword_removal(words)
		counts = count_words(stop_removals)
		import math

		prior_animasi_2d = (priors["animasi_2d"] / sum(priors.values()))
		prior_animasi_3d = (priors["animasi_3d"] / sum(priors.values()))
		prior_desain_multimedia = (priors["desain_multimedia"] / sum(priors.values()))
		prior_basis_data = (priors["basis_data"] / sum(priors.values()))
		prior_pemodelan_perangkat_lunak = (priors["pemodelan_perangkat_lunak"] / sum(priors.values()))
		prior_pemrograman_dasar = (priors["pemrograman_dasar"] / sum(priors.values()))
		prior_komputer_terapan_jaringan = (priors["komputer_terapan_jaringan"] / sum(priors.values()))
		prior_komunikasi_data = (priors["komunikasi_data"] / sum(priors.values()))
		prior_sistem_operasi_jaringan = (priors["sistem_operasi_jaringan"] / sum(priors.values()))

		log_prob_animasi_2d = 0.0
		log_prob_animasi_3d = 0.0
		log_prob_desain_multimedia = 0.0
		log_prob_basis_data = 0.0
		log_prob_pemodelan_perangkat_lunak = 0.0
		log_prob_pemrograman_dasar = 0.0
		log_prob_komputer_terapan_jaringan = 0.0
		log_prob_komunikasi_data = 0.0
		log_prob_sistem_operasi_jaringan = 0.0

		for w, cnt in list(counts.items()):
			# skip words that we haven't seen before, or words less than 3 letters long
			if w not in vocab or len(w) <= 3:
				continue

			p_word = vocab[w] / sum(vocab.values())
			p_w_given_animasi_2d = word_counts["animasi_2d"].get(w, 0.0) / sum(word_counts["animasi_2d"].values())
			p_w_given_animasi_3d = word_counts["animasi_3d"].get(w, 0.0) / sum(word_counts["animasi_3d"].values())
			p_w_given_desain_multimedia = word_counts["desain_multimedia"].get(w, 0.0) / sum(word_counts["desain_multimedia"].values())
			p_w_given_basis_data = word_counts["basis_data"].get(w, 0.0) / sum(word_counts["basis_data"].values())
			p_w_given_pemodelan_perangkat_lunak = word_counts["pemodelan_perangkat_lunak"].get(w, 0.0) / sum(word_counts["pemodelan_perangkat_lunak"].values())
			p_w_given_pemrograman_dasar = word_counts["pemrograman_dasar"].get(w, 0.0) / sum(word_counts["pemrograman_dasar"].values())
			p_w_given_komputer_terapan_jaringan = word_counts["komputer_terapan_jaringan"].get(w, 0.0) / sum(word_counts["komputer_terapan_jaringan"].values())
			p_w_given_komunikasi_data = word_counts["komunikasi_data"].get(w, 0.0) / sum(word_counts["komunikasi_data"].values())
			p_w_given_sistem_operasi_jaringan = word_counts["sistem_operasi_jaringan"].get(w, 0.0) / sum(word_counts["sistem_operasi_jaringan"].values())
			
			if p_w_given_animasi_2d > 0:
				log_prob_animasi_2d += math.log(cnt * p_w_given_animasi_2d / p_word)
			if p_w_given_animasi_3d > 0:
				log_prob_animasi_3d += math.log(cnt * p_w_given_animasi_3d / p_word)
			if p_w_given_desain_multimedia > 0:
				log_prob_desain_multimedia += math.log(cnt * p_w_given_desain_multimedia / p_word)
			if p_w_given_basis_data > 0:
				log_prob_basis_data += math.log(cnt * p_w_given_basis_data / p_word)
			if p_w_given_pemodelan_perangkat_lunak > 0:
				log_prob_pemodelan_perangkat_lunak += math.log(cnt * p_w_given_pemodelan_perangkat_lunak / p_word)
			if p_w_given_pemrograman_dasar > 0:
				log_prob_pemrograman_dasar += math.log(cnt * p_w_given_pemrograman_dasar / p_word)
			if p_w_given_komputer_terapan_jaringan > 0:
				log_prob_komputer_terapan_jaringan += math.log(cnt * p_w_given_komputer_terapan_jaringan / p_word)
			if p_w_given_komunikasi_data > 0:
				log_prob_komunikasi_data += math.log(cnt * p_w_given_komunikasi_data / p_word)
			if p_w_given_sistem_operasi_jaringan > 0:
				log_prob_sistem_operasi_jaringan += math.log(cnt * p_w_given_sistem_operasi_jaringan / p_word)

		rate = { 
			"animasi_2d": (log_prob_animasi_2d + math.log(prior_animasi_2d)),
			"animasi_2d": (log_prob_animasi_3d + math.log(prior_animasi_3d)),
			"desain_multimedia": (log_prob_desain_multimedia + math.log(prior_desain_multimedia)),
			"basis_data": (log_prob_basis_data + math.log(prior_basis_data)),
			"pemodelan_perangkat_lunak": (log_prob_pemodelan_perangkat_lunak + math.log(prior_pemodelan_perangkat_lunak)),
			"pemrograman_dasar": (log_prob_pemrograman_dasar + math.log(prior_pemrograman_dasar)),
			"komputer_terapan_jaringan": (log_prob_komputer_terapan_jaringan + math.log(prior_komputer_terapan_jaringan)),
			"komunikasi_data": (log_prob_komunikasi_data + math.log(prior_komunikasi_data)),
			"sistem_operasi_jaringan": (log_prob_sistem_operasi_jaringan + math.log(prior_sistem_operasi_jaringan))
		}
		
		predict = max(rate.iteritems(), key=operator.itemgetter(1))[0]
		result['predict'] = predict
		if(result['predict'] == result['target']):
			result['value'] = 1
		else:
			result['value'] = 0

		response.append(result)

	true_class = 0
	false_class = 0
	for res in response:
		if res['value'] == 0:
			false_class += 1
		else:
			true_class += 1

	data = {
		"akurasi": (float(true_class) / (true_class + false_class)) * 100,
		"datas": response
	}

	# return jsonify(data)
	# result = []
	# result.append((animasi_2d))
	return render_template('klasifikasi_custom.html', result=data)

@app.route('/test_sixexamples', methods=['GET'])
def test_sixexamples():
	vocab = {}
	word_counts = {
		"animasi_2d": {},
		"animasi_3d": {},
		"desain_multimedia": {},
		"basis_data": {},
		"pemodelan_perangkat_lunak": {},
		"pemrograman_dasar": {},
		"komputer_terapan_jaringan": {},
		"komunikasi_data": {},
		"sistem_operasi_jaringan": {}
	}
	priors = {
		"animasi_2d": 0.,
		"animasi_3d": 0.,
		"desain_multimedia": 0.,
		"basis_data": 0.,
		"pemodelan_perangkat_lunak": 0.,
		"pemrograman_dasar": 0.,
		"komputer_terapan_jaringan": 0.,
		"komunikasi_data": 0.,
		"sistem_operasi_jaringan": 0.
	}
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
		# ok time to start counting stuff...
		priors[category] += 1
		text = open(f).read()
		words = tokenize(text)
		stopword_removals = stopword_removal(words)
		# stem = stemming(stopword_removals)
		counts = count_words(stopword_removals)
		for word, count in list(counts.items()):
			# if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
			if word not in vocab:
				vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
			if word not in word_counts[category]:
				word_counts[category][word] = 0.0
			vocab[word] += count
			word_counts[category][word] += count

	response = []
	for f in find("examples-6"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"

		result = {}
		result['target'] = category
		result['file'] = f
		new_doc = open(f).read()
		words = tokenize(new_doc)
		stop_removals = stopword_removal(words)
		counts = count_words(stop_removals)
		import math

		prior_animasi_2d = (priors["animasi_2d"] / sum(priors.values()))
		prior_animasi_3d = (priors["animasi_3d"] / sum(priors.values()))
		prior_desain_multimedia = (priors["desain_multimedia"] / sum(priors.values()))
		prior_basis_data = (priors["basis_data"] / sum(priors.values()))
		prior_pemodelan_perangkat_lunak = (priors["pemodelan_perangkat_lunak"] / sum(priors.values()))
		prior_pemrograman_dasar = (priors["pemrograman_dasar"] / sum(priors.values()))
		prior_komputer_terapan_jaringan = (priors["komputer_terapan_jaringan"] / sum(priors.values()))
		prior_komunikasi_data = (priors["komunikasi_data"] / sum(priors.values()))
		prior_sistem_operasi_jaringan = (priors["sistem_operasi_jaringan"] / sum(priors.values()))

		log_prob_animasi_2d = 0.0
		log_prob_animasi_3d = 0.0
		log_prob_desain_multimedia = 0.0
		log_prob_basis_data = 0.0
		log_prob_pemodelan_perangkat_lunak = 0.0
		log_prob_pemrograman_dasar = 0.0
		log_prob_komputer_terapan_jaringan = 0.0
		log_prob_komunikasi_data = 0.0
		log_prob_sistem_operasi_jaringan = 0.0

		for w, cnt in list(counts.items()):
			# skip words that we haven't seen before, or words less than 3 letters long
			if w not in vocab or len(w) <= 3:
				continue

			p_word = vocab[w] / sum(vocab.values())
			p_w_given_animasi_2d = word_counts["animasi_2d"].get(w, 0.0) / sum(word_counts["animasi_2d"].values())
			p_w_given_animasi_3d = word_counts["animasi_3d"].get(w, 0.0) / sum(word_counts["animasi_3d"].values())
			p_w_given_desain_multimedia = word_counts["desain_multimedia"].get(w, 0.0) / sum(word_counts["desain_multimedia"].values())
			p_w_given_basis_data = word_counts["basis_data"].get(w, 0.0) / sum(word_counts["basis_data"].values())
			p_w_given_pemodelan_perangkat_lunak = word_counts["pemodelan_perangkat_lunak"].get(w, 0.0) / sum(word_counts["pemodelan_perangkat_lunak"].values())
			p_w_given_pemrograman_dasar = word_counts["pemrograman_dasar"].get(w, 0.0) / sum(word_counts["pemrograman_dasar"].values())
			p_w_given_komputer_terapan_jaringan = word_counts["komputer_terapan_jaringan"].get(w, 0.0) / sum(word_counts["komputer_terapan_jaringan"].values())
			p_w_given_komunikasi_data = word_counts["komunikasi_data"].get(w, 0.0) / sum(word_counts["komunikasi_data"].values())
			p_w_given_sistem_operasi_jaringan = word_counts["sistem_operasi_jaringan"].get(w, 0.0) / sum(word_counts["sistem_operasi_jaringan"].values())
			
			if p_w_given_animasi_2d > 0:
				log_prob_animasi_2d += math.log(cnt * p_w_given_animasi_2d / p_word)
			if p_w_given_animasi_3d > 0:
				log_prob_animasi_3d += math.log(cnt * p_w_given_animasi_3d / p_word)
			if p_w_given_desain_multimedia > 0:
				log_prob_desain_multimedia += math.log(cnt * p_w_given_desain_multimedia / p_word)
			if p_w_given_basis_data > 0:
				log_prob_basis_data += math.log(cnt * p_w_given_basis_data / p_word)
			if p_w_given_pemodelan_perangkat_lunak > 0:
				log_prob_pemodelan_perangkat_lunak += math.log(cnt * p_w_given_pemodelan_perangkat_lunak / p_word)
			if p_w_given_pemrograman_dasar > 0:
				log_prob_pemrograman_dasar += math.log(cnt * p_w_given_pemrograman_dasar / p_word)
			if p_w_given_komputer_terapan_jaringan > 0:
				log_prob_komputer_terapan_jaringan += math.log(cnt * p_w_given_komputer_terapan_jaringan / p_word)
			if p_w_given_komunikasi_data > 0:
				log_prob_komunikasi_data += math.log(cnt * p_w_given_komunikasi_data / p_word)
			if p_w_given_sistem_operasi_jaringan > 0:
				log_prob_sistem_operasi_jaringan += math.log(cnt * p_w_given_sistem_operasi_jaringan / p_word)

		rate = { 
			"animasi_2d": (log_prob_animasi_2d + math.log(prior_animasi_2d)),
			"animasi_2d": (log_prob_animasi_3d + math.log(prior_animasi_3d)),
			"desain_multimedia": (log_prob_desain_multimedia + math.log(prior_desain_multimedia)),
			"basis_data": (log_prob_basis_data + math.log(prior_basis_data)),
			"pemodelan_perangkat_lunak": (log_prob_pemodelan_perangkat_lunak + math.log(prior_pemodelan_perangkat_lunak)),
			"pemrograman_dasar": (log_prob_pemrograman_dasar + math.log(prior_pemrograman_dasar)),
			"komputer_terapan_jaringan": (log_prob_komputer_terapan_jaringan + math.log(prior_komputer_terapan_jaringan)),
			"komunikasi_data": (log_prob_komunikasi_data + math.log(prior_komunikasi_data)),
			"sistem_operasi_jaringan": (log_prob_sistem_operasi_jaringan + math.log(prior_sistem_operasi_jaringan))
		}
		
		predict = max(rate.iteritems(), key=operator.itemgetter(1))[0]
		result['predict'] = predict
		if(result['predict'] == result['target']):
			result['value'] = 1
		else:
			result['value'] = 0

		response.append(result)

	true_class = 0
	false_class = 0
	for res in response:
		if res['value'] == 0:
			false_class += 1
		else:
			true_class += 1

	data = {
		"akurasi": (float(true_class) / (true_class + false_class)) * 100,
		"datas": response
	}

	# return jsonify(data)
	# result = []
	# result.append((animasi_2d))
	return render_template('klasifikasi_custom.html', result=data)

@app.route('/test_eightexamples', methods=['GET'])
def test_eightexamples():
	vocab = {}
	word_counts = {
		"animasi_2d": {},
		"animasi_3d": {},
		"desain_multimedia": {},
		"basis_data": {},
		"pemodelan_perangkat_lunak": {},
		"pemrograman_dasar": {},
		"komputer_terapan_jaringan": {},
		"komunikasi_data": {},
		"sistem_operasi_jaringan": {}
	}
	priors = {
		"animasi_2d": 0.,
		"animasi_3d": 0.,
		"desain_multimedia": 0.,
		"basis_data": 0.,
		"pemodelan_perangkat_lunak": 0.,
		"pemrograman_dasar": 0.,
		"komputer_terapan_jaringan": 0.,
		"komunikasi_data": 0.,
		"sistem_operasi_jaringan": 0.
	}
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
		# ok time to start counting stuff...
		priors[category] += 1
		text = open(f).read()
		words = tokenize(text)
		stopword_removals = stopword_removal(words)
		# stem = stemming(stopword_removals)
		counts = count_words(stopword_removals)
		for word, count in list(counts.items()):
			# if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
			if word not in vocab:
				vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
			if word not in word_counts[category]:
				word_counts[category][word] = 0.0
			vocab[word] += count
			word_counts[category][word] += count

	response = []
	for f in find("examples-8"):
		f = f.strip()
		if not f.endswith(".txt"):
			# skip non .txt files
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"

		result = {}
		result['target'] = category
		result['file'] = f
		new_doc = open(f).read()
		words = tokenize(new_doc)
		stop_removals = stopword_removal(words)
		counts = count_words(stop_removals)
		import math

		prior_animasi_2d = (priors["animasi_2d"] / sum(priors.values()))
		prior_animasi_3d = (priors["animasi_3d"] / sum(priors.values()))
		prior_desain_multimedia = (priors["desain_multimedia"] / sum(priors.values()))
		prior_basis_data = (priors["basis_data"] / sum(priors.values()))
		prior_pemodelan_perangkat_lunak = (priors["pemodelan_perangkat_lunak"] / sum(priors.values()))
		prior_pemrograman_dasar = (priors["pemrograman_dasar"] / sum(priors.values()))
		prior_komputer_terapan_jaringan = (priors["komputer_terapan_jaringan"] / sum(priors.values()))
		prior_komunikasi_data = (priors["komunikasi_data"] / sum(priors.values()))
		prior_sistem_operasi_jaringan = (priors["sistem_operasi_jaringan"] / sum(priors.values()))

		log_prob_animasi_2d = 0.0
		log_prob_animasi_3d = 0.0
		log_prob_desain_multimedia = 0.0
		log_prob_basis_data = 0.0
		log_prob_pemodelan_perangkat_lunak = 0.0
		log_prob_pemrograman_dasar = 0.0
		log_prob_komputer_terapan_jaringan = 0.0
		log_prob_komunikasi_data = 0.0
		log_prob_sistem_operasi_jaringan = 0.0

		for w, cnt in list(counts.items()):
			# skip words that we haven't seen before, or words less than 3 letters long
			if w not in vocab or len(w) <= 3:
				continue

			p_word = vocab[w] / sum(vocab.values())
			p_w_given_animasi_2d = word_counts["animasi_2d"].get(w, 0.0) / sum(word_counts["animasi_2d"].values())
			p_w_given_animasi_3d = word_counts["animasi_3d"].get(w, 0.0) / sum(word_counts["animasi_3d"].values())
			p_w_given_desain_multimedia = word_counts["desain_multimedia"].get(w, 0.0) / sum(word_counts["desain_multimedia"].values())
			p_w_given_basis_data = word_counts["basis_data"].get(w, 0.0) / sum(word_counts["basis_data"].values())
			p_w_given_pemodelan_perangkat_lunak = word_counts["pemodelan_perangkat_lunak"].get(w, 0.0) / sum(word_counts["pemodelan_perangkat_lunak"].values())
			p_w_given_pemrograman_dasar = word_counts["pemrograman_dasar"].get(w, 0.0) / sum(word_counts["pemrograman_dasar"].values())
			p_w_given_komputer_terapan_jaringan = word_counts["komputer_terapan_jaringan"].get(w, 0.0) / sum(word_counts["komputer_terapan_jaringan"].values())
			p_w_given_komunikasi_data = word_counts["komunikasi_data"].get(w, 0.0) / sum(word_counts["komunikasi_data"].values())
			p_w_given_sistem_operasi_jaringan = word_counts["sistem_operasi_jaringan"].get(w, 0.0) / sum(word_counts["sistem_operasi_jaringan"].values())
			
			if p_w_given_animasi_2d > 0:
				log_prob_animasi_2d += math.log(cnt * p_w_given_animasi_2d / p_word)
			if p_w_given_animasi_3d > 0:
				log_prob_animasi_3d += math.log(cnt * p_w_given_animasi_3d / p_word)
			if p_w_given_desain_multimedia > 0:
				log_prob_desain_multimedia += math.log(cnt * p_w_given_desain_multimedia / p_word)
			if p_w_given_basis_data > 0:
				log_prob_basis_data += math.log(cnt * p_w_given_basis_data / p_word)
			if p_w_given_pemodelan_perangkat_lunak > 0:
				log_prob_pemodelan_perangkat_lunak += math.log(cnt * p_w_given_pemodelan_perangkat_lunak / p_word)
			if p_w_given_pemrograman_dasar > 0:
				log_prob_pemrograman_dasar += math.log(cnt * p_w_given_pemrograman_dasar / p_word)
			if p_w_given_komputer_terapan_jaringan > 0:
				log_prob_komputer_terapan_jaringan += math.log(cnt * p_w_given_komputer_terapan_jaringan / p_word)
			if p_w_given_komunikasi_data > 0:
				log_prob_komunikasi_data += math.log(cnt * p_w_given_komunikasi_data / p_word)
			if p_w_given_sistem_operasi_jaringan > 0:
				log_prob_sistem_operasi_jaringan += math.log(cnt * p_w_given_sistem_operasi_jaringan / p_word)

		rate = { 
			"animasi_2d": (log_prob_animasi_2d + math.log(prior_animasi_2d)),
			"animasi_2d": (log_prob_animasi_3d + math.log(prior_animasi_3d)),
			"desain_multimedia": (log_prob_desain_multimedia + math.log(prior_desain_multimedia)),
			"basis_data": (log_prob_basis_data + math.log(prior_basis_data)),
			"pemodelan_perangkat_lunak": (log_prob_pemodelan_perangkat_lunak + math.log(prior_pemodelan_perangkat_lunak)),
			"pemrograman_dasar": (log_prob_pemrograman_dasar + math.log(prior_pemrograman_dasar)),
			"komputer_terapan_jaringan": (log_prob_komputer_terapan_jaringan + math.log(prior_komputer_terapan_jaringan)),
			"komunikasi_data": (log_prob_komunikasi_data + math.log(prior_komunikasi_data)),
			"sistem_operasi_jaringan": (log_prob_sistem_operasi_jaringan + math.log(prior_sistem_operasi_jaringan))
		}
		
		predict = max(rate.iteritems(), key=operator.itemgetter(1))[0]
		result['predict'] = predict
		if(result['predict'] == result['target']):
			result['value'] = 1
		else:
			result['value'] = 0

		response.append(result)
		
	true_class = 0
	false_class = 0
	for res in response:
		if res['value'] == 0:
			false_class += 1
		else:
			true_class += 1

	data = {
		"akurasi": (float(true_class) / (true_class + false_class)) * 100,
		"datas": response
	}

	# return jsonify(data)
	# result = []
	# result.append((animasi_2d))
	return render_template('klasifikasi_custom.html', result=data)

@app.route('/klasifikasi', methods=['GET'])
def klasifikasi():
	return render_template('klasifikasi.html')

@app.route('/get_train_data', methods=['GET'])
def get_train_data():
	docs = []
	for f in find("sample-data"):
		f = f.strip()
		if not f.endswith(".txt"):
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
	return render_template('index.html', data_trains =  docs)

@app.route('/get_test_data', methods=['GET'])
def get_test_data():
	docs = []
	for f in find("examples"):
		f = f.strip()
		if not f.endswith(".txt"):
			continue
		elif "animasi_2d" in f:
			category = "animasi_2d"
		elif "animasi_3d" in f:
			category = "animasi_3d"
		elif "desain_multimedia" in f:
			category = "desain_multimedia"
		elif "basis_data" in f:
			category = "basis_data"
		elif "pemodelan_perangkat_lunak" in f:
			category = "pemodelan_perangkat_lunak"
		elif "pemrograman_dasar" in f:
			category = "pemrograman_dasar"
		elif "komputer_terapan_jaringan" in f:
			category = "komputer_terapan_jaringan"
		elif "komunikasi_data" in f:
			category = "komunikasi_data"
		else:
			category = "sistem_operasi_jaringan"
		docs.append((category, f))
	return render_template('index.html', data_tests =  docs)

if __name__ == "__main__":
	app.run(debug=True)