import csv
import numpy as np

datalist = []

with open("train.csv", "r") as traindata:
    data = csv.reader(traindata)
    for row in data:
        datalist.append(row)

datalist = datalist[1:]

classlist = [float(i[2]) for i in datalist]
print(classlist[:5])

namelist = [float(len(i[3])) for i in datalist]
print(namelist[:5])

sexlist = [float({"male": 0, "female": 1}[i[4]]) for i in datalist]
print(sexlist[:5])

averageage = np.average([float(i[5]) for i in datalist if i[5] != ''])
print(averageage)

agelist = []

for i in datalist:
    if i[5] == "":
        agelist.append(averageage)
    else:
        agelist.append(float(i[5]))

print(agelist[:6])

sibsplist = [float(i[6]) for i in datalist]

print(sibsplist[:5])

parchlist = [float(i[7]) for i in datalist]

print(parchlist[:5])

ticketlist = []

for i in datalist:
    buffer = 0
    for j in i[8][::-1]:
        if j.isnumeric():
            buffer += 1
        else:
            break
    ticketlist.append(float(buffer))

print(ticketlist[:5])

farelist = [float(i[9]) for i in datalist]

print(farelist[:5])

Srider = []
Crider = []
Qrider = []

for i in datalist:
    if i[11] == "S":
        Srider.append(float(1))
        Crider.append(float(0))
        Qrider.append(float(0))
    elif i[11] == "C":
        Srider.append(float(0))
        Crider.append(float(1))
        Qrider.append(float(0))
    elif i[11] == "Q":
        Srider.append(float(0))
        Crider.append(float(0))
        Qrider.append(float(1))
    else:
        Srider.append(float(0))
        Crider.append(float(0))
        Qrider.append(float(0))

print(Srider[:5])
print(Crider[:5])
print(Qrider[:5])

epsilon = 0.001

clist = [1-2*float(i[1]) for i in datalist]
cvector = np.array(clist)
cepsilon = epsilon*cvector

vectorlist = [np.array([classlist[i], sexlist[i], agelist[i]]) for i in range(891)]

print(clist[:5])

print(vectorlist[:6])


classvector = epsilon*np.multiply(np.array(classlist),cvector)
namevector = epsilon*np.multiply(np.array(namelist),cvector)
sexvector = epsilon*np.multiply(np.array(sexlist),cvector)
agevector = epsilon*np.multiply(np.array(agelist),cvector)
sibspvector = epsilon*np.multiply(np.array(sibsplist),cvector)
parchvector = epsilon*np.multiply(np.array(parchlist),cvector)
ticketvector = epsilon*np.multiply(np.array(ticketlist),cvector)
farevector = epsilon*np.multiply(np.array(farelist),cvector)
Svector = epsilon*np.multiply(np.array(Srider),cvector)
Cvector = epsilon*np.multiply(np.array(Crider),cvector)
Qvector = epsilon*np.multiply(np.array(Qrider),cvector)

trainlist = [classvector, sexvector, agevector]

winit = [1 for i in range(3)]
binit = 1

def calc(w,b,pos):
    wxb = np.dot(w, vectorlist[pos])+b

    if clist[pos]*wxb > 75:
        return 0

    return 1/(np.exp(clist[pos]*wxb)+1)

def train(wstart,bstart,depth):
    w, b = wstart, bstart

    for i in range(depth):
        vallist = [calc(w,b,i) for i in range(len(vectorlist))]
        valvect = np.array(vallist)

        wchange = np.array([np.dot(valvect, trainlist[i]) for i in range(np.size(vectorlist[0]))])
        bchange = np.dot(valvect, cepsilon)

        if i%1000 == 0:
            #print(loss(w,b))
            print(f"{w}, {b}")
            print(np.sum(valvect))

        w = w + wchange
        b = b + bchange

    return [w+wchange, b+bchange]

modelvals = train(winit, binit, 15000)

print(modelvals)

wfinal = modelvals[0]
bfinal = modelvals[1]

def calcexec(w, b, pos):
    wxb = np.dot(w, vectorlist[pos]) + b

    if wxb > 75:
        return 0

    return 1 / (np.exp(wxb) + 1)

def modelcalc(person):
    if calcexec(wfinal, bfinal, person) > 0.5:
        return 1

    return 0

print([calcexec(wfinal, bfinal, i) for i in range(891) if min(1-calcexec(wfinal, bfinal, i), calcexec(wfinal, bfinal, i))>0.01])

#print(classvector)