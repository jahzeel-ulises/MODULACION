from collections import OrderedDict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import json
from collections import defaultdict

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nrclex import NRCLex # NRC Word-Emotion Association Lexicon (aka EmoLex) 

from memspectrum import MESA
from scipy.fft import fft, fftfreq

import gzip
import random

from scipy.signal import lfilter
from scipy import signal

from gensim.models import Word2Vec
from gensim.models import FastText

##########################################################

def text_clean(sent):
    import re
    from unidecode import unidecode
    import unicodedata as ud
    from nltk.tokenize import word_tokenize

    d = {ord('\N{combining acute accent}'):None}
    sent = ud.normalize('NFD',sent).translate(d)
    sent = re.sub(r"[^\w .,;?!\n]+", "", sent)
    sent = re.sub(r"[^\w .,;?!]", " ", sent)
    sent = re.sub(r"[^\w .,]", ".", sent)
    sent = re.sub(r"[^\w ]", "", sent)
    sent = re.sub(r"[0-9]", "", sent)
    sent = re.sub(r"[α-ωςϑϕϜϝϞϠϰϱ]", "", sent)
    sent = re.sub(r"[ΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]", "", sent)
    sent = re.sub(r"[A-Z]{2}", "", sent)
    return word_tokenize(sent)

""" 
========================
COMPUTE EMBEDDINGS
========================
"""
def compute_embeddings(df,columna_docs='docs',embed='w2v', lookUpT=False, vec_dim=100,normalized=False):
    from nltk.tokenize import word_tokenize
    import re
    import unicodedata as ud

    # import modules & set up logging
    if embed=='w2v':
        from gensim.models import Word2Vec as w2v
    else:
        from gensim.models import FastText as fstxt


    # obtain vocabulary word types     
    # types=df[columna_docs].str.split(' ', expand=True).stack().unique()
    #Obtenemos los documentos
    Textos = []
    text=df[columna_docs].values.tolist()

    for sent in text:
        d = {ord('\N{combining acute accent}'):None}
        s = ud.normalize('NFD',sent).translate(d)
        s = re.sub(r"[^\w .,;?!\n]+", "", s)
        s = re.sub(r"[^\w .,;?!]", " ", s)
        s = re.sub(r"[^\w .,]", ".", s)
        s = re.sub(r"[^\w ]", "", s)
        s = re.sub(r"[0-9]", "", s)
        s = re.sub(r"[α-ωςϑϕϜϝϞϠϰϱ]", "", s)
        s = re.sub(r"[ΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]", "", s)
        Textos.append(s)
    if lookUpT:
        types=[]
        for s in Textos:
            types += word_tokenize(s)
        types = np.unique(types)
        # Data Frame of vocabulary and word embeddings
        typesdf=pd.Series(types).to_frame()
        typesdf.rename(index=int,columns={0:'Palabra'},inplace=True)

        #Add Emebddings placeholders
        #Se necesita convertir el DF a diccionario
        #luego se agregan vectores de dimension N,
        #como registros nuevos del diccionario
        #para reconvertirlo en un DF de vuelta
        dico=typesdf.to_dict('records',into=OrderedDict)
        #Add real-valued embedding vectors
        for reg in dico:
            reg['vector']=np.zeros(vec_dim)
        typesdf=pd.DataFrame.from_dict(dico)
        typesdf.set_index('Palabra',inplace=True)

    #Creamos las oraciones, este será la entrada del modelo W2V
    documentos=[]
    for s in Textos:
        documentos.append(word_tokenize(s))
    if embed=='w2v':
        # Referencia: Word2Vec(documentos, min_count=1, workers=4, window=5)
        model = w2v(documentos, min_count=1, vector_size=vec_dim, workers=4, window=5)
    else:
        model = fstxt(documentos, vector_size=vec_dim, workers=4, window=5, alpha=0.025)
    if lookUpT:
        words = typesdf.index.values.tolist()
        if normalized:
            for w in words:
                if w in model.wv:
                    typesdf.at[w,'vector'] = model.wv[w]/np.linalg.norm(model.wv[w])
                else:
                    typesdf.at[w,'vector'] = float("nan")                
        else:
            for w in words:
                if w in model.wv:
                    typesdf.at[w,'vector'] = model.wv[w]
                else:
                    typesdf.at[w,'vector'] = float("nan")                
        return typesdf,model
    else:
        return model

""" 
========================
FFT PHASE FORMATTING
========================
""" 
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex)) 

    
""" 
========================
BUILD SIGNAL
========================
""" 
def build_signal(s,modelo,thrshld=5,norm=False,stpw=False):
    vec_size = len(modelo.wv.get_vector(list(modelo.wv.key_to_index)[0]))
    if not stpw:
        signal = np.array([])
    else:
        signal = []
    if isinstance(s,list):
        for w in s:
            if w in modelo.wv.index_to_key:
                if not stpw:
                    signal = np.concatenate((signal,modelo.wv.get_vector(w, norm=norm)))
                else:
                    signal.append(modelo.wv.get_vector(w, norm=norm))
        if not stpw:
            if len(signal) < thrshld*vec_size:
                return np.nan
        else:
            if len(signal) < thrshld:
                return np.nan    
    else:
        for w in word_tokenize(s):
            if w in modelo.wv.index_to_key:
                if not stpw:
                    signal = np.concatenate((signal,modelo.wv.get_vector(w, norm=norm)))
                else:
                    signal.append(modelo.wv.get_vector(w, norm=norm))
        if not stpw:
            if len(signal) < thrshld*vec_size:
                return np.nan
        else:
            if len(signal) < thrshld:
                return np.nan
    return signal

""" 
========================
BUILD DATA
========================
""" 
def build_data(df,col,modelo,norm=False,stopw=False,th=5,frac=1):
    print(f"Getting TEXT from column: {col}")
    df['signal'] = np.nan
    df['signal'] = df[col].apply(lambda x: build_signal(x,modelo,norm=norm,thrshld=th,stpw=stopw))

    df.dropna(inplace=True)

    df['TokenS'] = df[col].apply(lambda x: [token for token in word_tokenize(x) if token in modelo.wv.key_to_index])
    df.dropna(inplace=True)

    l1 = df.TokenS.values.tolist()
    l2 = [w for l in l1 for w in l ]
    wf_model = Counter(l2)

    df['TokenS'] = df.TokenS.apply(lambda x: [(w,wf_model[w]) for w in x])

    items = df.sample(frac=frac, random_state=1)
    items.reset_index(inplace=True, drop=True)
    
    return items


""" 
========================
GET SIGNAL
========================
""" 
def get_signal(df,row,norm=False,plot=False,stpw=False):
    from nltk.corpus import stopwords
    
    stop_words = set(stopwords.words('english'))
    out_words = []
    word_labels = []
    fonts = 10
    if 'signal' in df.columns:
        signal = df.iloc[row].signal
    else:
        print("Please, run BUILD_DATA() first...")
        return

    if 'TokenS' in df.columns:
        for w,wf in df.iloc[row]['TokenS']:
            out_words.append(w)
            word_labels.append((w,wf))
    else:
        print("Please, run BUILD_DATA() first...")
        return

    if not stpw:
        if plot:
            ax = plt.gca()
            fig = ax.get_figure()
            fig.set_size_inches(14,5)
            ax.plot(df.iloc[row].signal)
            print(f"signal.shape: {df.iloc[row].signal.shape}")
            offs = 50
            losTicks = [offs + i for i in partition_ticks(df.iloc[row].signal)]
            losTicks.insert(0,offs)
            # print(losTicks)
            for xc in partition_ticks(df.iloc[row].signal):
                plt.axvline(x=xc,c='k',ls='--',lw=1)
            k=0
            if len(word_labels)>9:
                fonts=8
            for label in word_labels:
                ax.annotate(str(label),
                            xy=(losTicks[k], ax.get_ylim()[1]/2),
                            xytext=(0, 40),  # offsets.
                            textcoords='offset points',
                            ha='center', va='bottom', fontsize=fonts)

                k+=1
            plt.show()
        if norm:
            signal /= np.linalg.norm(signal)
        return(signal,out_words)
    else:
        widx = [(i,w) for i,w in enumerate(out_words) if w in stop_words]
        # print(widx)
        stw = [w for i,w in widx]
        sti = [i for i,w in widx]
        texto = [w for w in out_words if w not in stw]
        stl = [label for label in word_labels if label[0] in texto]
        sig_stw = []
        signal_stpw = np.array([])
        for i in range(len(df.iloc[row].signal)):
            if i not in sti:
                sig_stw.append(df.iloc[row].signal[i])
        for v in sig_stw:
            # print(f"v.shape: {v.shape}")
            signal_stpw = np.concatenate((signal_stpw,v))
            # print(f"signal_stpw.shape: {signal_stpw.shape}")
        if plot:
            plt.plot(signal_stpw)
            fig.set_size_inches(14,5)
            ax = plt.gca()
            offs = 50
            losTicks = [offs + i for i in partition_ticks(signal_stpw)]
            losTicks.insert(0,offs)
            for xc in partition_ticks(signal_stpw):
                plt.axvline(x=xc,c='k',ls='--',lw=1)
            k=0
            if len(stl)>9:
                fonts=8
            for label in stl:
                ax.annotate(str(label),
                            xy=(losTicks[k], ax.get_ylim()[1]/2),
                            xytext=(0, 40),  # 4 points vertical offset.
                            textcoords='offset points',
                            ha='center', va='bottom',fontsize=fonts)

                k+=1
            plt.show()
        if norm:
            signal_stpw /= np.linalg.norm(signal_stpw)
        return(signal_stpw,texto)
    

""" 
===========================
GET SPECTRA - PREDICT DATA
===========================
""" 
def get_spectra_predict(data,method='maxent',plot=False,predict=True,fwdt=100):
    if method=='maxent':
        M = MESA()
        median = None
        p, a_k, _ = M.solve(data)
        N, dt = len(data), 1 /len(data)  
        spec, frequencies = M.spectrum(dt)
        spec = np.fft.fftshift(spec)
        frequencies = np.fft.fftshift(frequencies)
        if not predict:
            if plot:
                fig, axs = plt.subplots()
                axs.plot(spec,frequencies)
                plt.show()
        else:
            if plot:
                fig, axs = plt.subplots(2)
                axs[0].plot(spec,frequencies)
            time = np.arange(0,N)
            M.solve(data[:-fwdt]) 
            forecast = M.forecast(data[:-fwdt], length = fwdt, number_of_simulations = 1000, include_data = False) 
            if np.issubdtype(type(forecast), np.floating):
                if forecast == 0.0: 
                    # print("ERROR: ¡NO SE PUEDEN HACER PREDICCIONES CON ESTOS DATOS! se retorna NaN")
                    return median,(spec,frequencies)
            median = np.median(forecast, axis = 0) #Ensemble median 
            p5, p95 = np.percentile(forecast, (5, 95), axis = 0) #90% credibility boundaries

            if plot:            
                axs[1].plot(time[:-fwdt], data[:-fwdt], color = 'k')
                axs[1].fill_between(time[-fwdt:], p5, p95, color = 'b', alpha = .5, label = '90% Cr.') 
                axs[1].plot(time[-fwdt:], data[-fwdt:], color = 'k', linestyle = '-.', label = 'Observed data') 
                axs[1].plot(time[-fwdt:], median, color = 'r', label = 'median estimate')
                plt.show()
        return median,(spec,frequencies)
    else:
        # Number of samples in normalized_tone
        # N = int(SAMPLE_RATE * DURATION)
        fig, axs = plt.subplots(2,figsize=(10, 6))

        N, dt = len(data), 1 /len(data)  
        yf = fft(data)
        xf = fftfreq(N,dt)
        yf = np.fft.fftshift(yf)
        xf = np.fft.fftshift(xf)
        
        fft_x = np.linspace(-N/2,N/2,N, True)
        fft_fwhl = np.fft.fft(data, norm='ortho')
        ampl_fft_fwhl = np.abs(fft_fwhl)

        #plt.axis('tight')
        #plt.xlim([0,200])
        if plot:
            axs[0].plot(xf,np.abs(yf))
            axs[1].scatter(xf, np.angle(yf),s=5)

            tau = np.pi
            den = 2
            major = Multiple(den, tau)
            minor = Multiple(den*4, tau)
            axs[1].grid(True)
            #axs[1].set_aspect(1.0)
            #axs[1].axhline(0, color='black', lw=2)
            #axs[1].axvline(0, color='black', lw=2)
            axs[1].yaxis.set_major_locator(major.locator())
            axs[1].yaxis.set_minor_locator(minor.locator())
            axs[1].yaxis.set_major_formatter(major.formatter())
            plt.show()
        return xf,yf


""" 
========================
PREDICT WORD NEIGHBORS
========================
""" 
def predict_word_neighbors(we,modelo,real_w,top_n=10):
    
    neighbors = modelo.wv.most_similar([we], topn=top_n)
    true_neighbors = modelo.wv.most_similar(real_w, topn=top_n)
    
    return neighbors, true_neighbors


""" 
========================
MAXENT EXAMPLE
========================
""" 
def maxent_example(filter='FIR',notch=False):
    N, dt = 1000, .01  #Number of samples and sampling interval
    time = np.arange(0, N) * dt
    frequency = 2  
    noisysignal = np.sin(2 * np.pi * frequency * time) + np.random.normal(.4, size = 1000) 

    fig,axs =plt.subplots(3,2,figsize=(12,5)) # ,figsize=(10,10)
    axs[0,0].plot(time, noisysignal, color = 'k')

    # LPF
    if filter == 'FIR':
        n = 15  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
    else:
        b, a = signal.butter(3, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, noisysignal, zi=zi*noisysignal[0])

    #filtsignal = lfilter(b, a, noisysignal)
    filtsignal = signal.filtfilt(b, a, noisysignal)
    if notch:
        fs = 1/dt  # Sample frequency (Hz)
        f0 = 0.1  # Frequency to be removed from signal (Hz)
        Q = 1.0  # Quality factor

        # Design notch filter
        b, a = signal.iirnotch(f0, Q, fs)
        filtsignal = signal.filtfilt(b, a, filtsignal)

    axs[0,1].plot(filtsignal, linewidth=2, linestyle="-", c="b")  # smooth by filter

    # spectrum by MAXENT
    M1 = MESA() 
    M1.solve(noisysignal) 
    spectrum, frequencies = M1.spectrum(dt)  #Computes on sampling frequencies 
    spectrum = np.fft.fftshift(spectrum)
    frequencies = np.fft.fftshift(frequencies)
    #user_frequencies = np.linspace(1.5, 2.5)
    #user_spectrum = M.spectrum(dt, user_frequencies) #Computes on desired frequency grid
    axs[1,0].plot(spectrum, frequencies)
    #plt.plot(user_frequencies,user_spectrum)
    axs[1,0].set_xlim([-10,10])

    M2 = MESA() 
    M2.solve(filtsignal) 
    spectrum, frequencies = M2.spectrum(dt)  #Computes on sampling frequencies 
    spectrum = np.fft.fftshift(spectrum)
    frequencies = np.fft.fftshift(frequencies)
    #user_frequencies = np.linspace(1.5, 2.5)
    #user_spectrum = M.spectrum(dt, user_frequencies) #Computes on desired frequency grid
    axs[1,1].plot(spectrum, frequencies)
    #plt.plot(user_frequencies,user_spectrum)
    axs[1,1].set_xlim([-10,10])

    # spectrum by FFT
    fft_x = np.linspace(0,N*dt,N)
    fft_fwhl = np.fft.fft(noisysignal, norm='ortho')
    ampl_fft_fwhl = np.abs(fft_fwhl)
    axs[2,0].plot(fft_x*N*dt, ampl_fft_fwhl,alpha=0.6)
    axs[2,0].set_xlim([0,10])


    fft_x = np.linspace(0,N*dt,N)
    fft_fwhl = np.fft.fft(filtsignal, norm='ortho')
    ampl_fft_fwhl = np.abs(fft_fwhl)
    axs[2,1].plot(fft_x*N*dt, ampl_fft_fwhl,alpha=0.6)
    axs[2,1].set_xlim([0,10])


""" 
========================
MAXENT PREDICT EXAMPLE
========================
""" 

def maxent_example_predict(fwdt=200):
    N, dt = 1000, .01  #Number of samples and sampling interval
    time = np.arange(0, N) * dt
    frequency = 2  
    noisysignal = np.sin(2 * np.pi * frequency * time) + np.random.normal(.4, size = 1000) 


    # LPF
    n = 15  # the larger n is, the smoother curve will be
    bf = [1.0 / n] * n
    af = 1

    b, a = signal.butter(3, 0.05)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, noisysignal, zi=zi*noisysignal[0])

    filtsignalB = signal.filtfilt(b, a, noisysignal)
    filtsignalFIR = signal.filtfilt(bf, af, noisysignal)

    datos = [noisysignal, filtsignalB, filtsignalFIR]

    M = MESA()

    time = np.arange(0,N)
    for data in datos:
        p, a_k, _ = M.solve(data)
        median = None
        spec, frequencies = M.spectrum(dt)
        spec = np.fft.fftshift(spec)
        frequencies = np.fft.fftshift(frequencies)
        M.solve(data[:-fwdt]) 
        forecast = M.forecast(data[:-fwdt], length = fwdt, number_of_simulations = 1000, include_data = False) 
        median = np.median(forecast, axis = 0) #Ensemble median 
        p5, p95 = np.percentile(forecast, (5, 95), axis = 0) #90% credibility boundaries

        plt.figure()
        _, ax = plt.subplots(2)
        ax[0].plot(spec,frequencies)
        ax[1].plot(time[:-fwdt], data[:-fwdt], color = 'k')
        ax[1].fill_between(time[-fwdt:], p5, p95, color = 'b', alpha = .5, label = '90% Cr.') 
        ax[1].plot(time[-fwdt:], data[-fwdt:], color = 'k', linestyle = '-.', label = 'Observed data') 
        ax[1].plot(time[-fwdt:], median, color = 'r', label = 'median estimate')


""" 
========================
MEAN SPECTRUM FOR TEXT
========================
""" 
def mean_spectrum_for_text(df,modelo,text_col=None,min_len=5,stw=False):
    df_aux = df.copy()
    if 'signal' not in df_aux.columns:
        if text_col == None:
            print("building signal column, but no <text_col> parameter given...")
            return
        else:
            print(f"building signal column with '{text_col}' as text column name")
            print(f"stop_words == {stw}")
            print(f"minimum text length == {min_len}")
            df_aux = wes.build_data(df_aux,text_col,modelo,stopw=stw,th=min_len)
    data = df_aux.signal.values.tolist()
    dat_norms = []
    dat_vecs = []
    M = MESA()
    vec_size = len(modelo.wv.get_vector(list(modelo.wv.key_to_index)[0]))
    size_limit = vec_size*min_len
    for vec in data:
        # v = vec/np.linalg.norm(vec)
        v = vec
        p, a_k, _ = M.solve(v)
        N, dt = len(v), 1 /len(v)  
        x, y = M.spectrum(dt)
        dat_norms.append(np.linalg.norm(y))
        dat_vecs.append(y[:size_limit])
    
    m = np.argmax(np.asarray(dat_norms))
    n = np.argmin(np.asarray(dat_norms))
    min_spectrum = np.asarray(dat_vecs)[n]
    max_spectrum = np.asarray(dat_vecs)[m]
    mean_spectrum = np.mean(np.asarray(dat_vecs).reshape(size_limit,-1),axis=1) 
    return mean_spectrum,min_spectrum,max_spectrum


""" 
========================
FILTER DATA SIGNAL
========================
""" 
def filter_data(data,type='FIR',coef=10):
    fig,axs =plt.subplots(3,2,figsize=(12,5))

    if type != "FIR":
        # Filtro IIR
        b, a = signal.butter(3, 0.05)
    else:
        # Filtro FFR
        n = coef  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1


    filtdata = signal.filtfilt(b, a, data)
    axs[0,0].plot(data)
    axs[0,1].plot(filtdata)

    N, dt = len(data), 1 /len(data)
    # spectrum by MAXENT
    M1 = MESA() 
    M1.solve(data) 
    spectrum, frequencies = M1.spectrum(dt)  #Computes on sampling frequencies 
    #user_frequencies = np.linspace(1.5, 2.5)
    #user_spectrum = M.spectrum(dt, user_frequencies) #Computes on desired frequency grid
    axs[1,0].plot(spectrum, frequencies)
    #plt.plot(user_frequencies,user_spectrum)
    axs[1,0].set_xlim([0,100])

    M2 = MESA() 
    M2.solve(filtdata) 
    spectrum, frequencies = M2.spectrum(dt)  #Computes on sampling frequencies 
    #user_frequencies = np.linspace(1.5, 2.5)
    #user_spectrum = M.spectrum(dt, user_frequencies) #Computes on desired frequency grid
    axs[1,1].plot(spectrum, frequencies)
    #plt.plot(user_frequencies,user_spectrum)
    axs[1,1].set_xlim([0,100])

    # spectrum by FFT
    fft_x = np.linspace(0,N*dt,N)
    fft_fwhl = np.fft.fft(data, norm='ortho')
    ampl_fft_fwhl = np.abs(fft_fwhl)
    axs[2,0].plot(fft_x*N*dt, ampl_fft_fwhl,alpha=0.6)
    axs[2,0].set_xlim([0,.1])


    fft_x = np.linspace(0,N*dt,N)
    fft_fwhl = np.fft.fft(filtdata, norm='ortho')
    ampl_fft_fwhl = np.abs(fft_fwhl)
    axs[2,1].plot(fft_x*N*dt, ampl_fft_fwhl,alpha=0.6)
    axs[2,1].set_xlim([0,.1])
    plt.show()


""" 
========================
FILTER FREQ RESPONSE
========================
""" 
def filter_freq_response(b, a, fs):
    freq, h = signal.freqz(b, a, fs=fs)
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([-25, 10])
    ax[0].grid(True)
    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, 100])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid(True)
    plt.show()


""" 
========================
PARTITION UTILITIES
========================
""" 
def partition (list_in, partes):
    list_out = []
    n = len(list_in)//partes
    resto = len(list_in)-n*partes
    if resto == 0:
        j=0
        for i in range(partes):
            list_out.append(list_in[j:n+j])
            j+=n
    else:
        j=0
        for i in range(partes):
            if i != partes-1:
                list_out.append(list_in[j:n+j])
                j+=n
            else:
                list_out.append(list_in[j:])
    return list_out

def partition_ticks (list_in, vec_size=100):
    n = len(list_in)//vec_size
    return [100*i for i in range(1,n)]


""" 
=======================================
DATOS DE POEMAS DE PROYECTO GUTENBERG
=======================================
""" 
def load_poems_gutenberg(data_path):
    with open(data_path+"Poems.json", "r") as f:
        poems = json.load(f)
    poems = [poem for poem in poems if len(poem["poem"].split()) < 100]
    print(len(poems))
    f.close()
    return poems

""" 
=======================================
CATEGORIZACION NRCLEX DE POEMAS
=======================================
""" 
def build_poems_dataset(data_path,emotion_list,poems):
    poem_emotions = defaultdict(list)
    missing = 0
    missing_poems = []


    # poems are read from the Gutenberg database and stored in the variable poems
    for i, poem in enumerate(poems):
        score = defaultdict(int)
        id = poem["id"]
        content = poem["poem"]

        # Creating a dictionary with frequencies of each of the emotions associated with a poem
        emotion = NRCLex(content).affect_frequencies

        # Removing degree of positivity and negativity from the dictionary as we are only concerned with emotions
        emotion.pop("positive")
        emotion.pop("negative")

        if len(emotion) == 0:
            # Missing indicates a poem that does not pertain to any particular emotion
            # Models for all emotion categories use these poems for training
            missing += 1
            missing_poems.append(poem)
        else:
            max_val = max(emotion.values())
            if list(emotion.values()).count(max_val) > 6:
                missing += 1
                missing_poems.append(poem)
            elif list(emotion.values()).count(max_val) > 1:
                pass
            else:
                poem_emotions[max(emotion, key=emotion.get)].append(poem)

        # if i % 1000 == 0:
        #     print(i)

    print("Number of Missing Poems:", missing)

    tot = 0
    for e in emotion_list:
        print("Number of poems belonging to {} = {}".format(i, len(poem_emotions[e])))
        tot += len(poem_emotions[e])
        with open(data_path+e+".json", "w") as outfile:
            json.dump(poem_emotions[e], outfile)

    with open(data_path+"missing.json", "w") as outfile:
        json.dump(missing_poems, outfile)

    print("Total number of poems in all categories: ", tot)

""" 
========================
TEST ANNOY
========================
""" 
def annoy_indexing_test(modelo):
    from gensim.similarities.annoy import AnnoyIndexer
    import time

    # Set up the model and vector that we are using in the comparison
    annoy_index = AnnoyIndexer(modelo, 100)

    normed_vectors = modelo.wv.get_normed_vectors()

    # Dry run to make sure both indexes are fully in RAM
    first_term = list(modelo.wv.key_to_index)[0]
    # equivalente a vector = modelo.wv.get_normed_vectors()[0]
    vector = modelo.wv.get_vector(first_term, norm=True)

    modelo.wv.most_similar([vector], topn=5, indexer=annoy_index)
    modelo.wv.most_similar([vector], topn=5)

    def avg_query_time(annoy_index=None, queries=1000):
        #Average query time of a most_similar method over 1000 random queries.
        total_time = 0
        for _ in range(queries):
            rand_vec = normed_vectors[np.random.randint(0, len(modelo.wv))]
            start_time = time.process_time()
            modelo.wv.most_similar([rand_vec], topn=5, indexer=annoy_index)
            total_time += time.process_time() - start_time
        return total_time / queries

    queries = 1000

    gensim_time = avg_query_time(queries=queries)
    annoy_time = avg_query_time(annoy_index, queries=queries)
    print("Gensim (s/query):\t{0:.5f}".format(gensim_time))
    print("Annoy (s/query):\t{0:.5f}".format(annoy_time))
    speed_improvement = gensim_time / annoy_time
    print ("\nAnnoy is {0:.2f} times faster on average on this particular run".format(speed_improvement))


