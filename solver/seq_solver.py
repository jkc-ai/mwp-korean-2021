import itertools
import numpy as np
from scipy import optimize
from itertools import combinations, permutations, product
import re

MAX_SEQ = 200
DEBUG = False
LOCAL = True
def make_seq(c,seq_type = 1):
    if seq_type ==2:
        return make_seq_poly(c)
    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    try:
        c3 = c[3]
    except:
        c3 = 0

    try:
        c4= c[4]
    except:
        c4 = 0

    seq=[]
    for i in range(MAX_SEQ):

        if i==0:
            seq.append(c0)
        elif i==1:
            seq.append(c1)
        else:

            seq.append(seq[i-2]*c2+seq[i-1]*c3)

    return seq


def make_seq_poly(c):
    c0 = c[0]
    c1 = c[1]
    try:
        c2 = c[2]
    except:
        c2 = 0
    try:
        c3 = c[3]
    except:
        c3 = 0


    seq=[]
    for i in range(MAX_SEQ):
        seq.append(c0+c1*i+c2*i**2+c3*i**3)
    return seq


def print_seq(c,seq_type=1):
    if LOCAL: print('predicted seq:',make_seq(c,seq_type)[:10])

def solve_seq_pattern(seq_inp, init=[1,1,1,0]):
    def cal_loss(c):
        seq_pred= make_seq(c)
        seq_pred = seq_pred[0:len(seq_inp)]
        loss = 0

        
        for i in range(len(seq_inp)):
            
            if seq_inp[i]>=0:
                loss = loss+ (seq_pred[i]-seq_inp[i])**2
                
        return loss

    def cal_loss_poly(c):
        seq_pred= make_seq_poly(c)
        seq_pred = seq_pred[0:len(seq_inp)]
        loss = 0

        
        for i in range(len(seq_inp)):
            
            if seq_inp[i]>=0:
                loss = loss+ (seq_pred[i]-seq_inp[i])**2
                
        return loss
    if LOCAL: print('1nd try: polynomical')

    n_seq = get_n_seq(seq_inp)
    x = init
    if len(x)>n_seq:
        x = x[0:n_seq]

    if len(x)>4:
        x = x[0:4]
    out = optimize.fmin(cal_loss_poly, x,xtol=1E-10,ftol=1E-20,maxiter=5000,full_output=True,disp=DEBUG)
    loss = out[1]
    if out[4]!=0:
        if LOCAL: print('max_iteration warning!(1)')
    if LOCAL: 
        print('1st loss:', loss)
        print('c:',out[0])
    seq_type = 2

    if loss > 1E-1:
        seq_type = 1
        x = init
        if len(x)>=4:
            x = x[0:4]
        out = optimize.fmin(cal_loss, x,xtol=1E-10,ftol=1E-20,maxiter=5000,full_output=True,disp=DEBUG)
        loss = out[1]
        if LOCAL: print('2nd loss',loss)
        if out[4]!=0:
            if LOCAL: print('max_iteration warning!(1)')

    out_c = out[0].tolist()
    if LOCAL: print("out_c:",out_c)
    if len(init)>n_seq:
        out_c.append(0)
    if LOCAL: print("out_c:",out_c)


    return out_c, loss, seq_type

def cal_seq(c,n,seq_type=1):
    seq = make_seq(c,seq_type)
    return seq[n-1]


def find_seq(seq_inp,c,eq,seq_type=1):
    seq_pred = make_seq(c,seq_type)
    n_seq=get_n_seq(seq_inp)
    c = c+[0,0,0]
    code = ""
    for i in range(len(seq_inp)):
        if seq_inp[i]==-1:
            A = seq_pred[i]
            code = code+ 'A = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'A' : n = i
        elif seq_inp[i]==-2:
            B = seq_pred[i]
            code = code+ 'B = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'B' : n = i
        elif seq_inp[i]==-3:
            C = seq_pred[i]
            code = code+ 'C = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'C' : n = i
        elif seq_inp[i]==-4:
            D = seq_pred[i]
            code = code+ 'D = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'D' : n = i
        elif seq_inp[i]==-5:
            X = seq_pred[i]
            code = code+ 'X = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'X' : n = i
        elif seq_inp[i]==-6:
            Y = seq_pred[i]
            code = code+ 'Y = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'Y' : n = i
        elif seq_inp[i]==-7:
            Z = seq_pred[i]
            code = code+ 'Z = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'Z' : n = i
    if LOCAL: print(eq)
    return eval(eq), code

def find_seq_string(seq,target):
    seq_ori = seq
    if LOCAL: print("find_seq_string:",target, seq)

    if seq[-1] < 0:
        seq = seq[:-1]

    code = ''
    code = code + "seq="+str(seq)+'\n'


    pattern_len = len(seq)
    key = 0
    for i, n in enumerate(seq):
        if i==0: key = seq[i]
        if i>1 and seq[i]==key:
            pattern_len = i
            break
    code = code + "pattern_len = len(seq)\n"
    if LOCAL: print(seq)
    if str(type(target))=="<class 'int'>":
        
        out = seq[(target-1)%pattern_len]
        
        code = code + "target=%d\n"%target
        code = code + "print(seq[(target-1)%pattern_len])"

    else:
        if target == 'A': 
            value = -1
        if target == 'B': 
            value = -2
        if target == 'C': 
            value = -3
        if target == 'D': 
            value = -4
        if target == 'X': 
            value = -5
        if target == 'Y': 
            value = -6
        if target == 'Z': 
            value = -7
        idx = seq_ori.index(value)
        out = seq_ori[idx%pattern_len]
        code = code + "print(seq[%d%%%d])"%(idx,pattern_len)
        
    if LOCAL: print(code)
    return out, code

def print_seq_eq(c,target,seq_type):
    out = ''

    if LOCAL: print('c:', c)
    c.append(0)
    c.append(0)

    if seq_type ==2:
        if str(type(target))=="<class 'str'>":
            if len(target)==1:
                n = len(target)
                print("warning!!!!")
                out = "print(int(round(%f+%f*%d+%f*%d**2+%f*%d**3)))"%(c[0],c[1],n,c[2],n,c[3],n)
            else:
                out = "print(int(round(%s)))\n"%target
        else:
            n = target-1
            out = "print(int(round(%f+%f*%d+%f*%d**2+%f*%d**3)))"%(c[0],c[1],n,c[2],n,c[3],n)

    elif seq_type ==1:
        out = out + 'c0 = %f\n'%c[0]
        out = out + 'c1 = %f\n'%c[1]
        out = out + 'c2 = %f\n'%c[2]
        out = out + 'c3 = %f\n'%c[3]
        out = out + 'c4 = %f\n'%c[4]
        out = out + 'seq=[]\n'
        out = out + 'for i in range(%d):\n'%50
        out = out + '    if i==0: seq.append(c0)\n'
        out = out + '    elif i==1: seq.append(c1)\n'
        out = out + '    else: seq.append(seq[i-2]*c2+seq[i-1]*c3)\n'

        if str(type(target))=="<class 'str'>":
            out = out + 'print(%s)'%target
        else:
            out = out + 'print(seq[%d])'%(target-1)
    return out

def find_index_string(seq, w):
    key = 0
    if w=='A': key = -1
    if w=='B': key = -2
    if w=='C': key = -3
    if w=='D': key = -4
    if w=='X': key = -5
    if w=='Y': key = -6
    if w=='Z': key = -7


    if key==0:
        return 0
    else:
        return seq.index(key)

def get_n_seq(seq):

    seq_new = [x for x in seq if x>=0]
    n_seq = len(seq_new)

    return n_seq

def seq_pred(seq_str,targets=[],eqs=''):
    if LOCAL: print('initial:', targets, eqs)
    seq_ori = seq_str

    seq_str = seq_str.replace('A', '-1')
    seq_str = seq_str.replace('B', '-2')
    seq_str = seq_str.replace('C', '-3')
    seq_str = seq_str.replace('D', '-4')
    seq_str = seq_str.replace('X', '-5')
    seq_str = seq_str.replace('Y', '-6')
    seq_str = seq_str.replace('Z', '-7')

    if LOCAL: print(seq_str)

    seq = eval(seq_str)
    target = None

    if len(targets)==1:
        target = targets[0]

    if str(type(seq[0]))=="<class 'str'>" :
        if LOCAL: print('string')
        return find_seq_string(seq,len(seq)+1)

    n_seq = get_n_seq(seq)
    if LOCAL: print("no of seq:", n_seq)
    c,loss,seq_type = solve_seq_pattern(seq, [seq[0],1,0,0,0])

    if LOCAL: print('targets=', targets)
    if str(type(target))=="<class 'str'>":
        if target.isdigit() == True:
            target = int(target)

    if len(targets)>1:
        if LOCAL: print('multiple target! output eq:',targets)
        code = ""
        for idx, tar in enumerate(targets):
            if idx==0:
                A = cal_seq(c,tar,seq_type)
                if LOCAL: print('A=',A)
                if seq_type == 2:
                    code = code +"A = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'A=%d\n'%A
            elif idx==1:
                B = cal_seq(c,tar,seq_type)
                if LOCAL: print('B=',B)
                if seq_type == 2:
                    code = code +"B = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'B=%d\n'%B
            elif idx==2:
                C = cal_seq(c,tar,seq_type)
                if LOCAL: print('C=',C)
                if seq_type == 2:
                    code = code +"C = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'C=%d\n'%C
            elif idx==3:
                D = cal_seq(c,tar,seq_type)
                if LOCAL: print('D=',D)
                if seq_type == 2:
                    code = code +"D = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'D=%d\n'%D

            elif idx==4:
                X = cal_seq(c,tar,seq_type)
                if LOCAL: print('X=',X)
                if seq_type == 2:
                    code = code +"X = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'X=%d\n'%X

            elif idx==5:
                Y = cal_seq(c,tar,seq_type)
                if LOCAL: print('Y=',Y)
                if seq_type == 2:
                    code = code +"Y = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'Y=%d\n'%Y
            elif idx==6:
                Z = cal_seq(c,tar,seq_type)
                if LOCAL: print('Z=',Z)
                if seq_type == 2:
                    code = code +"Z = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'Z=%d\n'%Z                        
                
        out = eval(eqs)
        if LOCAL: print('eqs:', eqs)
        if LOCAL: print(eqs, out)
        code = code + 'print(int(round(%s)))'%eqs
        return out, code

    if LOCAL: print('target:',target)
    if str(type(target))=="<class 'int'>": 
        if loss > 1:
            if LOCAL: print('solve by string pattern (int target)')
            return find_seq_string(seq,target)
        else:
            if LOCAL: print("simple seq")
            if LOCAL: print_seq(c,seq_type)
            return cal_seq(c,target,seq_type), print_seq_eq(c,target,seq_type)
    else:
        if LOCAL: print("case of equation output")
        if loss > 1:
            if LOCAL: print('solve by string pattern(string target')
            return find_seq_string(seq,eqs)
        else:
            if LOCAL: print_seq(c,seq_type)
            
            out, code = find_seq(seq,c,eqs,seq_type)
            
            index = find_index_string(seq,eqs)
            if index ==0:
                return out, code+ print_seq_eq(c,eqs,seq_type)
            else:
                return out, code+ print_seq_eq(c,index+1,seq_type)

## find variable by optimization...
def solve(eq):
    eq = '(('+eq+'))**2'
    eq = eq.replace('=',')-(')
    if LOCAL: print(eq)

    def cal_loss(x):
        out = eval(eq)
        return out

    out = optimize.fmin(cal_loss, 0, xtol=0.00000001, ftol=0.00000001, maxiter=1500, full_output=True, disp=DEBUG)

    out = round(out[0][0],2)
    if LOCAL: print(out)

    return 


korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
special_char = '?.,_'
def delete_str(word, chars):

    for char in chars:
        word = word.replace(char,'')
    return word


def solve_seq(input):

    text_nokor= re.sub(korean, '_', input).strip()
    if LOCAL: print(text_nokor)

    words = re.findall(r"[\w']+", text_nokor)
    find_num = False
    seqs = []

    if LOCAL: print(words)

    for word in words:

        if word.isalnum() :
            if word.isdigit()==True:
                find_num = True
                seqs.append(word)
            else:
                n = input.index(word)
                if find_num == True or input[n+1] == ',':
                    find_num = True
                    seqs.append(word)

        if find_num == True:
            if word.isalnum() == False:
                word = word.split('_')[0]
                if word!='':
                    seqs.append(word)
                break

    if LOCAL: print("sequence list:",seqs)
    seq_str= ",".join(seqs)
    if LOCAL: print(seq_str)


    words = text_nokor.split(' ')
    eqs = ''

    targets = find_target_no(input)
    
    for word in words:
        word = delete_str(word, special_char)
        word = word.replace(' ','')

        if word!='':
            eqs = word

    if LOCAL: print("ans:", eqs)

    return seq_pred(seq_str, targets, eqs)


def find_target_no(inp):
    if '번 째' in inp:
        inp = inp.replace('번 째', '번째')
    elif not('번째' in inp):
        inp = inp.replace('째', '번째')
    inp = inp.replace('번째', ' 번째')
    
    if LOCAL: print(inp)
        
    words = inp.split(' ')
    targets = []
    target = 0
    for idx, word in enumerate(words):
        if '번째' in word:
            w = words[idx-1]
            if '첫' in w:
                target = 1
            elif '두' in w:
                target = 2
            elif '세' in w:
                target = 3
            else:
                target = int(w)
            targets.append(target)

    if LOCAL: print(targets)
    return targets
    

def seq_solver(question:str, local = False):
    global LOCAL
    LOCAL = local

    ans, code = solve_seq(question)
    ans = int(round(ans))
    if local:
        print('ans:',ans)
        print(code)
    return { 'answer': ans, 'equation': code}


if __name__ == "__main__":
    q_list = ["주어진 숫자가 31, A, 33, 34, 35, B, 37, 38 일 경우, B-A에 해당하는 알맞은 수는 무엇일까요?",
                "2, 4, 8, 14, 22 에서 7번째에 올 수를 구하시오.",
                "1, 17, 33, 49, 65와 같은 규칙에서 25번째 놓일 수와 40번째 놓일 수를 각각 A와 B라 할 때, B-A를 구하시오.",
                "주어진 숫자가 31, A, 33, 34, 35, B, 37, 38 일 경우, B-A에 해당하는 알맞은 수는 무엇일까요?",
                "2, 4, 8, 14, 22 에서 7번째에 올 수를 구하시오.",
                "1, 17, 33, 49, 65와 같은 규칙에서 25번째 놓일 수와 40번째 놓일 수를 각각 A와 B라 할 때, B-A를 구하시오.",
                "주어진 숫자가 31, A, 33, 34, 35, B, 37, 38 일 경우, B에 해당하는 알맞은 수는 무엇일까요?",
                "1,2,3,4,5,6,7,1,2,3,4,5,6,7과 같이 반복되는 수열이 있습니다. 왼쪽에서 57번째 숫자는 무엇입니까?",
                "1, 5, 14, 30, 55, 91과 같은 규칙으로 수를 배열하고 있습니다. 9번째 수는 무엇입니까?",
                "자연수를 규칙에 따라 4, 7, 10, A, 16, 19로 배열하였습니다. A에 알맞은 수를 구하시오."]
    for i, q in enumerate(q_list):
        a = seq_solver(q, False)['answer']
        print(f"{i+1:2d} 번째 문제\n    - {'문제':2s}: {q}\n    - {'답':^3s}: {a}\n")
