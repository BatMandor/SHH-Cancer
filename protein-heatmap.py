#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#written in pymol command line, to create the 3d heatmap of the importance of the position of SHH mutation in survival rate 
#I changed the b factor values of the alpha carbons to the importance found in the regression model, and then used that to create a heat map of importance 
prot="3M1N"
inFile = open("/Users/rosenewkey-burden/Downloads/newBFactors.txt", 'r')
stored.newB = []
for line in inFile.readlines(): stored.newB.append( float(line) )
inFile.close()
alter "3M1N",  b=0.0
alter "3M1N" and n. CA, b=stored.newB.pop(0)
cmd.save("3M1N_newBFactors.pdb", "3M1N")
iterate (all), print (b) #checking if the output was correct
minval = min(val_list)
print minval
maxval = max(val_list)
print maxval
cmd.spectrum("b", "blue_white_red", "%s and n. CA"%prot, minimum=0, maximum=maxval)
cmd.ramp_new("ramp_obj", prot, range=[0, minval, maxval], color="[blue, white, red ]")
cmd.save("%s_newBFactors.pdb"%prot, "%s"%prot)
set opaque_background, 0
png /Users/rosenewkey-burden/Downloads/shh-heatmap.png, 0, 0, -1, ray=1

