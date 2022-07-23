from keystrokes_recorder import record, record_number
import pandas
import numpy


def time_based():
    while True:
        rec = record(10)
        df = pandas.DataFrame(numpy.transpose(numpy.array(rec)))
        print(df)


def keystrokes_based():
    while True:
        rec = record_number(100)
        df = pandas.DataFrame(numpy.transpose(numpy.array(rec)))
        print(df)


keystrokes_based()


# 1. Na czas, czy na znaki, bo jeżeli będzie na znaki to trzeba ustalić odpowiednią liczbę tych znaków, a jeżeli na
#   czas, to może się pojawić taki moment, kiedy nie będzie tych znaków dużo i co wtedy zrobimy. Trzeba to ustalić też.
# 2. Ile danych mamy zapisywać, generalnie im więcej tym lepiej, ale wszystkie informacje można obliczyć z dwóch
#   danych, dlugosc nacisniecia klawisza i odstep pomiedzy dwoma klawiszami, ale gosc na biometrii robil jakies dzikie
#   manewry z tymi rzeczami
# 3. Jaka dokładność zapisu czasu
# 4. Gdzie zapisywać modele ludzi, jaka będzie baza danych,
#   trzeba to wszystko znaleźć
# 5. Możemy przyjąć model taki, że nagrywanie przez x sekund, chyba że jest mniej niż y
#   znaków, to wtedy przejście do drugiego trybu.
