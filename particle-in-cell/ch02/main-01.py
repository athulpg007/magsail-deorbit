from world import World
from species import Species
import numpy as np

world=World(21,21,21)
world.setTime(2E-10,100)
world.setExtents(-0.1, -0.1, 0.0 , 0.1, 0.1, 0.2)

species1 = Species("O+", 16*world.AMU, world.QE, world)
species2 = Species("e-", world.ME, -1.0*world.QE, world)

world.addSpeciesList([species1,species2])

species1.loadParticlesBoxQS(world.x0, world.xm, 1E11, [41, 41, 41])
species2.loadParticlesBoxQS(world.x0, world.xc, 1E11, [21, 21, 21])

species1.computeNumberDensity()
species2.computeNumberDensity()

world.computeChargeDensity()

world.potentialSolver(10000, 1E-3)

world.efSolver()

for i in range(1,101):
    species1.advance()
    species2.advance()

    species1.computeNumberDensity()
    species2.computeNumberDensity()

    world.computeChargeDensity()

    world.potentialSolver(10000, 1E-4)

    world.efSolver()

    print("ts = "+str(i)+", nO+: "+str(len(species1.particleList))+", ne-: "+str(len(species2.particleList)))
    
    if i%100==0:
        x_arr1 = np.zeros(len(species1.particleList))
        y_arr1 = np.zeros(len(species1.particleList))
        z_arr1 = np.zeros(len(species1.particleList))

        x_arr2 = np.zeros(len(species2.particleList))
        y_arr2 = np.zeros(len(species2.particleList))
        z_arr2 = np.zeros(len(species2.particleList))

        for j in np.arange(0,len(species1.particleList)):
            pos = species1.particleList[j].pos
            x_arr1[j] = pos[0]
            y_arr1[j] = pos[1]
            z_arr1[j] = pos[2]

        for j in np.arange(0,len(species2.particleList)):
            pos = species2.particleList[j].pos
            x_arr2[j] = pos[0]
            y_arr2[j] = pos[1]
            z_arr2[j] = pos[2]
            
        np.savetxt('species_1_x_'+str(i).zfill(4)+'.txt', x_arr1)
        np.savetxt('species_1_y_'+str(i).zfill(4)+'.txt', y_arr1)
        np.savetxt('species_1_z_'+str(i).zfill(4)+'.txt', z_arr1)
        
        np.savetxt('species_2_x_'+str(i).zfill(4)+'.txt', x_arr2)
        np.savetxt('species_2_y_'+str(i).zfill(4)+'.txt', y_arr2)
        np.savetxt('species_2_z_'+str(i).zfill(4)+'.txt', z_arr2)
