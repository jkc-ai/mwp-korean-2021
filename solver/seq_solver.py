import itertools
import numpy as np
from scipy import optimize
from itertools import combinations, permutations, product

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
        # if i==0:
        #     seq.append(c0)
        # elif i==1:
        #     seq.append(seq[i-1]*c2+c1)
        # elif i==2:
        #     seq.append(seq[i-2]*c3+seq[i-1]*c2+c1)
        # else:
        #     seq.append(seq[i-3]*c4+ seq[i-2]*c3+seq[i-1]*c2+c1)
        if i==0:
            seq.append(c0)
        elif i==1:
            seq.append(c1)
        else:
        # elif i==2:
            # seq.append(seq[i-2]*c2+seq[i-1]*c3+c4)
            seq.append(seq[i-2]*c2+seq[i-1]*c3)
        # else:
            # seq.append(seq[i-3]*c5+ seq[i-2]*c3+seq[i-1]*c2+c1)

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

    # c4 = c[4]

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

        # print(seq_pred, seq_inp)
        for i in range(len(seq_inp)):
            # loss = loss+ abs(seq_pred[i]-seq_inp[i])
            if seq_inp[i]>=0:
                loss = loss+ (seq_pred[i]-seq_inp[i])**2
                # loss = loss+ abs(seq_pred[i]-seq_inp[i])
        return loss

    def cal_loss_poly(c):
        seq_pred= make_seq_poly(c)
        seq_pred = seq_pred[0:len(seq_inp)]
        loss = 0

        # print(seq_pred, seq_inp)
        for i in range(len(seq_inp)):
            # loss = loss+ abs(seq_pred[i]-seq_inp[i])
            if seq_inp[i]>=0:
                loss = loss+ (seq_pred[i]-seq_inp[i])**2
                # loss = loss+ abs(seq_pred[i]-seq_inp[i])
        return loss
    # print(cal_loss([0,0,0]))
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

    if loss > 1E-1: ## iteration 타입 시도
        seq_type = 1
        x = init
        if len(x)>=4:
            x = x[0:4]
        out = optimize.fmin(cal_loss, x,xtol=1E-10,ftol=1E-20,maxiter=5000,full_output=True,disp=DEBUG)
        # print(out)
        loss = out[1]
        if LOCAL: print('2nd loss',loss)
        if out[4]!=0:
            if LOCAL: print('max_iteration warning!(1)')

    # out_c = np.around(out[0])
    out_c = out[0].tolist()
    if LOCAL: print("out_c:",out_c)
    if len(init)>n_seq:
        out_c.append(0)
    if LOCAL: print("out_c:",out_c)

    # return out_c.astype('int64'), loss, seq_type
    return out_c, loss, seq_type

def cal_seq(c,n,seq_type=1):
    seq = make_seq(c,seq_type)
    return seq[n-1]


def find_seq(seq_inp,c,eq,seq_type=1):
    seq_pred = make_seq(c,seq_type)
    # n = 0
    n_seq=get_n_seq(seq_inp)
    # if n_seq==3: ## 수가 3개인 경우... c[3]추가..
    # c.append((0,0,0))
    c = c+[0,0,0]
    code = ""
    for i in range(len(seq_inp)):
        if seq_inp[i]==-1:
            A = seq_pred[i]
            # code = code+ 'A = %d\n'%A
            # out = "print(%d+%d*%d+%d*%d**2+%d*%d**3)"%(c[0],c[1],n,c[2],n,c[3],n)
            code = code+ 'A = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'A' : n = i
        elif seq_inp[i]==-2:
            B = seq_pred[i]
            # code = code+ 'B = %d\n'%B
            code = code+ 'B = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'B' : n = i
        elif seq_inp[i]==-3:
            C = seq_pred[i]
            # code = code+ 'C = %d\n'%C
            code = code+ 'C = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'C' : n = i
        elif seq_inp[i]==-4:
            D = seq_pred[i]
            # code = code+ 'C = %d\n'%C
            code = code+ 'D = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'D' : n = i
        elif seq_inp[i]==-5:
            X = seq_pred[i]
            # code = code+ 'C = %d\n'%C
            code = code+ 'X = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'X' : n = i
        elif seq_inp[i]==-6:
            Y = seq_pred[i]
            # code = code+ 'C = %d\n'%C
            code = code+ 'Y = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'Y' : n = i
        elif seq_inp[i]==-7:
            Z = seq_pred[i]
            # code = code+ 'C = %d\n'%C
            code = code+ 'Z = %f+%f*%d+%f*%d**2+%f*%d**3\n'%(c[0],c[1],i,c[2],i,c[3],i)
            if eq == 'Z' : n = i
    if LOCAL: print(eq)
    return eval(eq), code

# def repeats(string):
#     for x in range(1, len(string)):
#         substring = string[:x]

#         if substring * (len(string)//len(substring))+(substring[:len(string)%len(substring)]) == string:
#             if LOCAL: print(substring)
#             return "break"

#     if LOCAL: print(string)

def find_seq_string(seq,target):
    seq_ori = seq
    if LOCAL: print("find_seq_string:",target, seq)
    # seq_str = ''.join(seq)
    # pattern = repeats(seq_str)
    # pattern = seq
    # pattern_len = len(pattern)
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
        # seq_len = len(pattern)
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
        # return seq[idx]
        
        # return 
        # print(idx)
    if LOCAL: print(code)
    return out, code

def print_seq_eq(c,target,seq_type):
    out = ''
    # n=target
    # if str(type(target))=="<class 'str'>":

    #     out = "print(%s)\n"%target
    if LOCAL: print('c:', c)
    # if len(c)==3:
    c.append(0)
    c.append(0)

    if seq_type ==2:
        # print("c0+c1*i+c2*i**2+c3*i**3"%(c0,c1,)
        if str(type(target))=="<class 'str'>":
            if len(target)==1:
                n = len(target) ## 잘못됨..
                # n = find_index_string(seq, w):
                print("warning!!!!")
                out = "print(int(round(%f+%f*%d+%f*%d**2+%f*%d**3)))"%(c[0],c[1],n,c[2],n,c[3],n)
            else:
                out = "print(int(round(%s)))\n"%target
        else:
            n = target-1
            out = "print(int(round(%f+%f*%d+%f*%d**2+%f*%d**3)))"%(c[0],c[1],n,c[2],n,c[3],n)
        # return out
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
        # out = out + '    elif i==1: seq.append(seq[i-1]*c2+c1)\n'
        out = out + '    else: seq.append(seq[i-2]*c2+seq[i-1]*c3)\n'
        # out = out + '    elif i==2: seq.append(seq[i-2]*c3+seq[i-1]*c2+c1)\n'
        # out = out + '    else: seq.append(seq[i-3]*c4+ seq[i-2]*c3+seq[i-1]*c2+c1)\n'

        if str(type(target))=="<class 'str'>":
            out = out + 'print(%s)'%target
        else:
            out = out + 'print(seq[%d])'%(target-1)
    
    
    return out
    # c0 = c[0]
    # c1 = c[1]
    # c2 = c[2]
    # c3 = c[3]
    # c4 = c[4]  

    # print('n:', n)
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
    # seq_str = seq_str.replace('E', '-5')
    seq_str = seq_str.replace('X', '-5')
    seq_str = seq_str.replace('Y', '-6')
    seq_str = seq_str.replace('Z', '-7')

    if LOCAL: print(seq_str)

    seq = eval(seq_str)
    target = None
    # if targets==[]:
    #     target = len(seq)+1
        # print('n=',n)
    if len(targets)==1:
        target = targets[0]

    if str(type(seq[0]))=="<class 'str'>" :
        if LOCAL: print('string') ## 요소가 String 타입인 경우 -> 패턴으로..
        return find_seq_string(seq,len(seq)+1)

    # c,loss,seq_type = solve_seq_pattern(seq, [seq[0],1,1,0,0])

    #seq에 숫자가 몇개인가에 따라 초기화 다르게...
    # seq_new = [x for x in seq if x>=0]
    n_seq = get_n_seq(seq)
    if LOCAL: print("no of seq:", n_seq)
    # if n_seq==3:
    #     # if LOCAL: 
    #     c,loss,seq_type = solve_seq_pattern(seq, [seq[0],1,0])
    # else:
    c,loss,seq_type = solve_seq_pattern(seq, [seq[0],1,0,0,0])
    
    #print(type(c))
    
    # make_seq(c)
    # print('loss',loss)
    # print_seq(c)



    # print (str(type(target)))
    if LOCAL: print('targets=', targets)
    # if str(type(n))=="<class 'int'>": ## 
    if str(type(target))=="<class 'str'>":
        if target.isdigit() == True:
            target = int(target)

    if len(targets)>1:  ## target이 2개 이상인 경우... 
        if LOCAL: print('multiple target! output eq:',targets)
        # out, code = find_seq2(seq,c,targets,seq_type)
        code = ""
        for idx, tar in enumerate(targets):
            if idx==0:  ## 야매.... A,B,C 순서로 나온다고 가정
                A = cal_seq(c,tar,seq_type)
                if LOCAL: print('A=',A)
                if seq_type == 2:
                    # code = code +"A = %d+%d*%d+%d*%d**2+%d*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
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
                C = cal_seq(c,tar,seq_type)
                if LOCAL: print('D=',D)
                if seq_type == 2:
                    code = code +"D = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'D=%d\n'%D

            elif idx==4:
                C = cal_seq(c,tar,seq_type)
                if LOCAL: print('X=',X)
                if seq_type == 2:
                    code = code +"X = %f+%f*%d+%f*%d**2+%f*%d**3\n"%(c[0],c[1],tar-1,c[2],tar-1,c[3],tar-1)
                else:
                    code = code +'X=%d\n'%X

            elif idx==5:
                C = cal_seq(c,tar,seq_type)
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

    # print('eqs:', eqs)
    # if len(targets)==1:

    # if eqs!='':
    if LOCAL: print('target:',target)
    if str(type(target))=="<class 'int'>": ## 
        if loss > 1: ## 에러가 높은 경우 반복문구로
            if LOCAL: print('solve by string pattern (int target)')
            return find_seq_string(seq,target) # 전체 스트링 반복이 아니라 패턴을 찾아야할 수도?
        else:
            if LOCAL: print("simple seq")
            if LOCAL: print_seq(c,seq_type)
            return cal_seq(c,target,seq_type), print_seq_eq(c,target,seq_type)
    else:
        if LOCAL: print("case of equation output")
        if loss > 1:
            if LOCAL: print('solve by string pattern(string target')
            return find_seq_string(seq,eqs) # 전체 스트링 반복이 아니라 패턴을 찾아야할 수도?
        else:
            # print('targets:', eqs)
            if LOCAL: print_seq(c,seq_type)
            # if len(targets)==1:
            # print('output eq:',eqs)
            out, code = find_seq(seq,c,eqs,seq_type)
            
            index = find_index_string(seq,eqs)
            if index ==0:
                return out, code+ print_seq_eq(c,eqs,seq_type)
            else:
                return out, code+ print_seq_eq(c,index+1,seq_type)
            # elif len(targets)>1:
    # if str(type(target))=="<class 'int'>": ## target이 int인경우... (~번째?)
    # else: ## target이 string인 경우 "EX) A는?"
          


def solve(eq):
    eq = '(('+eq+'))**2'
    eq = eq.replace('=',')-(')
    if LOCAL: print(eq)

    def cal_loss(x):
        out = eval(eq)
        # print(out)
        return out

    out = optimize.fmin(cal_loss, 0, xtol=0.00000001, ftol=0.00000001, maxiter=1500, full_output=True, disp=DEBUG)

    out = round(out[0][0],2)
    if LOCAL: print(out)

    return 


import re

korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
special_char = '?.,_'
def delete_str(word, chars):
    # words_new = []
    # for word in words:
    for char in chars:
        word = word.replace(char,'')
        # words_new.append(word)
    return word

    # import re
    # DATA = "Hey, you - what are you doing here!?"
    # print re.findall(r"[\w']+", DATA)
    # Prints ['Hey', 'you', 'what', 'are', 'you', 'doing', 'here']


def solve_seq(input):

    text_nokor= re.sub(korean, '_', input).strip()
    if LOCAL: print(text_nokor)

    # words = text_nokor.split(',')
    words = re.findall(r"[\w']+", text_nokor)
    find_num = False
    seqs = []

    if LOCAL: print(words)



    # find sequence
    for word in words:
        # word = word.replace('_', '')
        # word = word.split('_')[0]
        # words_new.append(word)
        # print(word)
        if word.isalnum() :
            if word.isdigit()==True: ## 숫자인 경우는 ㅇㅋ
                find_num = True
                seqs.append(word)
            else:
                n = input.index(word)
                # if input[n]==',' and find_num == True: ## 다음 문자가 쉼표..
                # if find_num == True or input[n:
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

    ## find output

    words = text_nokor.split(' ')
    eqs = ''

    targets = find_target_no(input)
    # if len(targets)==0:
    for word in words:
        # words_new.append(word)
        word = delete_str(word, special_char)
        word = word.replace(' ','')
        # print(word)
        # if word.isalnum() :
        if word!='':
            eqs = word
                # print('target:',target)

    # print('target:',targets,eqs)
    if LOCAL: print("ans:", eqs)

    return seq_pred(seq_str, targets, eqs)


def find_target_no(inp):
    if '번 째' in inp:
        inp = inp.replace('번 째', '번째')
    # print(inp)
    elif not('번째' in inp):
        inp = inp.replace('째', '번째')
    # if '번째' in inp:
    inp = inp.replace('번째', ' 번째')
    
    # elif '째' in inp:
    if LOCAL: print(inp)
        
    words = inp.split(' ')
    # print(w)
    targets = []
    target = 0
    for idx, word in enumerate(words):
        if '번째' in word:## or '째' in word:
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
    

def seq_solver(question:str, local):
    global LOCAL
    LOCAL = local

    ans, code = solve_seq(question)
    ans = int(round(ans))
    if local:
        print('ans:',ans)
        print(code)
    return { 'answer': ans, 'equation': code}


if __name__ == "__main__":
# execute only if run as a script
    # input = "주어진 숫자가 31, A, 33, 34, 35, B, 37, 38 일 경우, B-A에 해당하는 알맞은 수는 무엇일까요?"
    input = "2, 4, 8, 14, 22 에서 7번째에 올 수를 구하시오."
    # input = "1, 17, 33, 49, 65와 같은 규칙에서 25번째 놓일 수와 40번째 놓일 수를 각각 A와 B라 할 때, B-A를 구하시오."
    # input = "주어진 숫자가 31, A, 33, 34, 35, B, 37, 38 일 경우, B-A에 해당하는 알맞은 수는 무엇일까요?"
    # input = "2, 4, 8, 14, 22 에서 7번째에 올 수를 구하시오."
    # input = "1, 17, 33, 49, 65와 같은 규칙에서 25번째 놓일 수와 40번째 놓일 수를 각각 A와 B라 할 때, B-A를 구하시오."
    # input = "주어진 숫자가 31, A, 33, 34, 35, B, 37, 38 일 경우, B에 해당하는 알맞은 수는 무엇일까요?"
    # input = "1, 3, 5, 1, 3, 5, A 에서 반복되는 규칙에 따라 A에 알맞은 수는?"
    # input = "1,2,3,4,5,6,7,1,2,3,4,5,6,7과 같이 반복되는 수열이 있습니다. 왼쪽에서 57번째 숫자는 무엇입니까?"
    # input = "왼쪽부터 흰색 공 1개, 노란색 공 2개, 빨간색 공 3개가 반복되어 놓여 있습니다. 58번째 공의 색깔을 쓰시오."
    # input = "100개의 사탕을 태형, 남준, 윤기 3명에게 순서대로 두 개씩 나누어 줍니다. 91번째 사탕을 받는 사람은 누구입니까?"
    # input = "1, 5, 14, 30, 55, 91과 같은 규칙으로 수를 배열하고 있습니다. 9번째 수는 무엇입니까?"
    # input = "자연수를 규칙에 따라 4, 7, 10, A, 16, 19로 배열하였습니다. A에 알맞은 수를 구하시오."

    # print(round(285.000002))

    # out = seq_solver(input, True)
    # print(out['answer'])

    dataset_file = "../../../datasets_class/수학문제AI - 5. 규칙.tsv"


    import csv
    
    f = open(dataset_file, 'r', encoding='utf-8')
    rdr = csv.reader(f, delimiter='\t')

    # rdr = csv.reader(f)
    ok = 0
    no = 0
    err = 0
    # line = rdr[11]
    # print(line[0])
    # out = seq_solver(rdr[11][0], True)
    # quit()
    import time

    max_time = 0
    for idx, line in enumerate(rdr):
        t1 = time.time()
        # if idx == 108:
        #     # if idx == 107=: break
        #     print(idx, line[0])  
        #     seq_solver(line[0], True)
        # else:
        #     continue
        # if idx < 106:
            # continue
        # print(idx, line)
        
        # out = seq_solver(line[0], True)
        try:
            out = seq_solver(line[0],False)
        except Exception as e: 
            print(idx, line)  
            print(e)
            err = err +1
            # break
            continue

        # print(out['answer'],line[1])
        if abs (float(out['answer'])-float(line[1])) < 1E-5:
            ok = ok +1
        else:
            no = no +1
            print(idx, line)
            print(out['answer'],line[1])
            # break
            # out = seq_solver(line[0], True)

        # print(ok, no, err)
        t2 = time.time()
        e_t = t2-t1
        if max_time>e_t: max_time = e_t
        print('elapsed time:',e_t)
    f.close()
    print('max elapsed time:',max_time)
    print('정답:%d, 오답: %d, 오류: %d, 정확도: %f'%(ok, no, err, (ok/(ok+no+err))))
    # global LOCAL
    # LOCAL = False
    # ans, code = solve_seq(input)

    # print(ans)
    # print(code)

    # print(out.answer)
    # print(out.equation)
    ## ~번째... 경우만 따로 처리...
    ## 계산한 후에 A=

# A = 1.000000+16.000000*24+0.000000*24**2+-0.000000*24**3
# B = 1.000000+16.000000*39+0.000000*39**2+-0.000000*39**3
# print(int(round(B-A)))