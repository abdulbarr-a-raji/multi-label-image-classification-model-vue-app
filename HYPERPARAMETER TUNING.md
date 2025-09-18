HYPERPARAMETER TUNING
testing correct answers: [strawberry, kiwi] [orange, kiwi] [strawberry]

1.    opt=adam, lr=0.001, b=8, e=2, shuffle=on
      results: loss=0.552, bin_acc=0.812
      testing: [strawberry] [strawberry, orange, kiwi] [strawberry]
      score: 7/9

2.    opt=adam, lr=0.001, b=8, e=6, shuffle=on
      results: loss=0.214, bin_acc=0.990
      testing: [strawberry] [strawberry, orange, kiwi] [strawberry]
      score: 7/9

3.    opt=adam, lr=0.001, b=16, e=2, shuffle=on
      results: loss=0.618, bin_acc=0.781
      testing: [strawberry] [strawberry, kiwi] [none]
      score: 3/9

4.    opt=adam, lr=0.01, b=8, e=2, shuffle=on
      results: loss=0.603, bin_acc=0.698
      testing: [none] [orange, kiwi] [none]
      score: 3/9

5.    opt=rmsprop, lr=0.001, b=8, e=2, shuffle=on
      results: loss=0.608, bin_acc=0.698
      testing: [strawberry, kiwi] [strawberry, orange, kiwi] [strawberry, orange, kiwi]
      score: 6/9

6.    opt=adam, lr=0.001, b=8, e=2, shuffle=off
      results: loss=0.558, bin_acc=0.771
      testing: [strawberry, orange, kiwi] [strawberry, orange, kiwi] [strawberry, orange]
      score: 6/9

7.    opt=adam, lr=0.01, b=4, e=2, shuffle=on
      results: loss=0.599, bin_acc=0.740
      testing: [strawberry] [strawberry, orange] [strawberry, orange]
      score: 5/9

8.    opt=rmsprop, lr=0.01, b=8, e=10, shuffle=on
      results: loss=0.235, bin_acc=1.000
      testing: [strawberry] [strawberry, orange, kiwi] [strawberry]
      score: 7/9

9.    opt=rmsprop, lr=0.01, b=8, e=7, shuffle=on
      results: loss=0.235, bin_acc=1.000
      testing: [strawberry] [strawberry, orange, kiwi] [strawberry]
      score: 7/9
      seems to be optimum