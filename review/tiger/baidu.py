import re


def max_word(s):
    s = s.replace('\\', ' ')
    s = s.replace(',', ' ')
    s = s.replace('.', ' ')
    s = s.replace("\'", ' ', 2)
    print(s)

    pass

if __name__ == '__main__':
    s = "Look to the skies above London and you’ll see the usual suspects rainclouds, plane and pigeons. But by the end of the year, you might just see something else."
    max_word(s)
