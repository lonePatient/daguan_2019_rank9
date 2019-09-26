#encoding:utf-8
import re
import jieba

class ChinesePreProcessor(object):
    def __init__(self,min_len = 2,stopwords_path = None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.reset()

    def jieba_cut(self,sentence):
        '''
        jieba分词
        :param sentence:
        :return:
        '''
        seg_list = jieba.cut(sentence,cut_all=False)
        return ' '.join(seg_list)

    def reset(self):
        '''
        加载停用词
        :return:
        '''
        if self.stopwords_path:
            with open(self.stopwords_path,'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1

    def clean_length(self,sentence):
        '''
        去除长度小于min_len的文本
        :param sentence:
        :return:
        '''
        if len([x for x in sentence]) >= self.min_len:
            return sentence


    def full2half(self,sentence):
        '''
        全角转化为半角
        :param sentence:
        :return:
        '''
        ret_str = ''
        for i in sentence:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    def remove_stopword(self,sentence):
        '''
        去除停用词
        :param sentence:
        :return:
        '''
        words = sentence.split()
        x = [word for word in words if word not in self.stopwords]
        return " ".join(x)

    def get_china(self,sentence):
        '''
        获取中文
        :param sentence:
        :return:
        '''
        zhmodel = re.compile("[\u4e00-\u9fa5]")
        words = [x for x in sentence if zhmodel.search(x)]
        return ''.join(words)

    def remove_numbers(self,sentence):
        '''
        移除数字
        :param sentence:
        :return:
        '''
        words = sentence.split()
        x = [re.sub('\d+','',word) for word in words]
        return ' '.join([w for w in x if w !=''])

    def remove_whitespace(self,sentence):
        '''
        移除空白部分
        :param sentence:
        :return:
        '''
        x = ''.join([x for x in sentence if x !=' ' or x !='' or x!='  '])
        return x

    def __call__(self, sentence):
        x = sentence.strip('\n')
        x = self.full2half(x)
        # x = self.jieba_cut(x)
        # if self.stopwords_path:
        #     x = self.remove_stopword(x)
        x = self.remove_whitespace(x)
        x = self.get_china(x)
        x = self.clean_length(x)

        return x
