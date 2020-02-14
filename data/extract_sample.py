import os

'''
Run this file to take the first 20 lines of each CSV file in this 
directory and output them to /samples/sample_*.csv
'''
for filename in os.listdir("./"):
    if filename.endswith(".csv"):
        with open(filename) as myfile:
            head = [next(myfile) for x in range(20)]
            with open('sample/sample_'+filename, 'w+') as newfile:
                for line in head:
                    newfile.write(line)
