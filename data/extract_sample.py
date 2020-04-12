import os

'''
Run this file to take the first 300 lines of each CSV file in this
directory and output them to /samples/sample_*.csv
'''
for filename in os.listdir("./"):
    if filename.endswith(".csv"):
        with open(filename) as myfile:
            try:
                head = [next(myfile) for x in range(300)]
                with open('sample/sample_'+filename, 'w+') as newfile:
                    for line in head:
                        newfile.write(line)
            except:
                with open('sample/sample_'+filename, 'w+') as newfile:
                    for line in head:
                        newfile.write(line)
