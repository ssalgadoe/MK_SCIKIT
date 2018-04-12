

ifilename = '../data_sets/creditcard.csv'
ofilename = '../data_sets/creditcard_sample.csv'
lines = 15000
in_file= open(ifilename,'r')
out_file= open(ofilename,'w')

counter = 0
done = False
while not done:
    line = in_file.readline()
    out_file.write(line)
    counter +=1
    if counter >= lines:
        done = True

in_file.close()
out_file.close()
print('done!')