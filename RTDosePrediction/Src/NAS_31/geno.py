import pickle

f = open("best_genotype.pkl", "wb")
a = ("Genotype(down=[(0.3699315, 'dep_conv'), (0.42127913, 'se_conv'), (0.3746939659118652, 'none'), (0.8690855741500855, 'down_dep_conv')],up=[(0.8095518, 'se_conv'), (0.73751825, 'se_conv'), (0.3311869382858276, 'none'), (0.6009042859077454, 'up_interpolate')])", 1)
pickle.dump(a,f)
f.close()


