outfilename = 'trainval_duke_1.txt'

with open('trainval_duke.txt') as f:
	lines = f.readlines()

outfile = open(outfilename,'w')

totalnum = len(lines)

skip = 100

for ind,line in enumerate(lines):

	if ind%skip==0:
		outfile.write(line)

outfile.close()
