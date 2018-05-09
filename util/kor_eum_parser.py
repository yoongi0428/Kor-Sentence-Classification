# -*- coding: utf-8 -*-
from util.freq_han import freq_han

special = "0123456789,;.!?:'""/\|_@#$%^&*~`+-=<>()[]{}ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ" # len = 80
hanguel_len = len(freq_han)
dic_han = {freq_han[k]: k for k in range(hanguel_len)}
special_dic = {}
for i,k in enumerate(special):
    special_dic[k] = i

def decompose_str_as_one_hot_eum(string):
    tmp_list = []
    for x in string:
        da = decompose_as_one_hot(ord(x))
        if da[0] == 9999:
            continue
        tmp_list.extend(da)
    return tmp_list

def decompose_as_one_hot(in_char):
    if chr(in_char) in freq_han:  # 음절 인코딩에 포함될때.
        result = dic_han[chr(in_char)]
    elif chr(in_char) in special:
        result = hanguel_len + special_dic[chr(in_char)]
    else:
        result = 9999 # unknown
    return [result]

if __name__ == "__main__":
    print(len(freq_han))


