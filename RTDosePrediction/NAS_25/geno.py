import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.2509537, 'conv'), (0.5414705, 'se_conv'), (0.32593646049499514, 'none'), (0.4626232981681824, 'down_dil_conv')] ,up=[(0.3285157, 'dep_conv'), (0.3065012, 'se_conv'), (0.22266273498535155, 'identity'), (0.32882168889045715, 'up_dil_conv')])", 1)
pickle.dump(a,f)
f.close()


