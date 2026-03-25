# -*- coding: utf-8 -*-
"""
@author: akbilay 
modified by cnevers

"""

import numpy as np
from io import StringIO
import os
import shutil

from joblib import Parallel, delayed, cpu_count

import vtk
import pyvista as pv

from tqdm import tqdm

from scipy import interpolate

########################################################################
#################################INPUTS#################################

wallAndTreeSurfaceFile = 'wallAndTreeSurfaces.stl'
#surface file to check if two positions see each other
#it runs faster if the stl file has fewer triangles

rMagMax = 100     #maximum ray distance when calculating view factors

#num_cores = 4
num_cores = cpu_count()

groundNormal = [0, 0, 1]    #vertical direction

########################################################################

def calculate(path, a, b, c, veg=False):

    timesteps = np.arange(a,b+c,c)

    s=open(path + '/system/air/probe_locs').read().replace('(','').replace(')','')
    pos = np.loadtxt(StringIO(s),ndmin=2)

    #For a certain timestep, each proc takes a position and calculated Tumrt. Then, they move onto the next timestep.
    #View factors are calculated only during the first timestep. They are written to a file and used for the following timesteps.
    for timestep_i, timestep in enumerate(timesteps):
        Parallel(n_jobs=num_cores)(delayed(calculate_parallel)(path, timesteps, timestep, timestep_i, veg, P_i, P, pos) for P_i, P in enumerate(tqdm(pos)))


    return

def calculate_parallel(path, timesteps, timestep, timestep_i, veg, P_i, P, pos):
    #i: person's position (side and top center points)
    #Ai: person's area (side and top area vectors)
    i, Ai = pedestrian_param(P)

    #read file Tambient
    Tambient_interp = read_files(path, 'Tambient', timesteps)
    #read file cloudCover
    cloudCover_interp = read_files(path, 'cloudCover', timesteps)
    #read file Idif
    Idif_interp = read_files(path, 'Idif', timesteps)

    #calculate view factors
    Fij, Fijsum, FijSky, FijsumSky = calculate_Fij(path, timesteps, P_i, i, Ai, veg)

    #calculate Tumrt (mean radiant temperature except the direct solar component) in parallel
    TumrtAvg = calculate_Tumrt(timestep, path, i, Ai, Tambient_interp, cloudCover_interp, Idif_interp, Fij, Fijsum, FijSky, FijsumSky, timestep_i)

    #write TumrtAvg
    if not os.path.exists(path + '/UTCI/'):
        os.makedirs(path + '/UTCI/')

    if not os.path.exists(path + '/UTCI/' + str(timestep) + '/'):
        try:
            os.makedirs(path + '/UTCI/' + str(timestep) + '/')
        except:
            pass

    if not os.path.exists(path + '/UTCI/' + str(timestep) + '/TumrtAvg'):
        f=open(path + '/UTCI/' + str(timestep) + '/TumrtAvg','w')
    else:
        f=open(path + '/UTCI/' + str(timestep) + '/TumrtAvg','a')
    a_ = np.hstack((timestep,P,TumrtAvg))
    np.savetxt(f,np.reshape(a_, ((1, a_.shape[0]))),fmt='%.4f')
    f.close()

def calculate_Fij(path, timesteps, P_i, i, Ai, veg):

    if not os.path.exists(path + '/UTCI/pos/' + str(P_i) + '/F.npz'):
        data = np.loadtxt(path + '/postProcessing/surfaces/' + str(timesteps[0]) + '/Sf_wallAndTreeSurfaces.raw',skiprows=2)
        j = data[:,0:3] #surface positions (center points)
        Aj = -1.* data[:,3:6] #surface areas (area vectors)

        dataSky = np.loadtxt(path + '/postProcessing/surfaces/' + str(timesteps[0]) + '/Sf_skySurfaces.raw',skiprows=2)
        jSky = dataSky[:,0:3] #surface positions (center points)
        AjSky = -1.* dataSky[:,3:6] #surface areas (area vectors) 

        checkImportedData(path, veg, j, jSky)

        Fij = np.zeros([len(Ai),len(j)])
        Fijsum = np.zeros([len(Ai),1])

        FijSky = np.zeros([len(Ai),len(jSky)])
        FijsumSky = np.zeros([len(Ai),1])

        #i: pedestrian position, Ai: pedestrian area vector
        #j: surface position, Aj: surface area vector

        #check if ray between pedestrian and surface (or sky) is intersected
        ray_end = j
        ray_endSky = jSky
        bool_is_intersected, bool_is_intersectedSky = is_intersected(i, ray_end, ray_endSky, wallAndTreeSurfaceFile)
        
        #wallAndTree view factors
        for n in range(len(i)):
            AiMag = np.linalg.norm(Ai[n,:])

            for m in range(len(j)):
                r = i[n,:]-j[m,:] #distance vector between the two positions i[n] and j[m]
                rMag = np.linalg.norm(r)

                if ((float(rMag) > 0) and (float(rMag) < rMagMax)):
                    if (bool_is_intersected[n][m]):
                        continue   #skip the remaining part if ray intersects walls

                    AjMag = np.linalg.norm(Aj[m,:])

                    ni = Ai[n,:]/AiMag #unit vector for Ai
                    nj = Aj[m,:]/AjMag #unit vector for Aj
                    cosThetaI = np.abs(np.dot(ni, r))/rMag
                    cosThetaJ = np.abs(np.dot(nj, r))/rMag

                    #test to make sure that the surfaces are looking towards each other
                    cosPhiTest = np.dot(Ai[n,:],i[n,:]-j[m,:]) #/ (np.linalg.norm(Ai[n,:])*np.linalg.norm(i[n,:]-j[m,:]))
                    if (float(cosPhiTest) < 0): #if the pedestrian surface is looking towards the other one
                        #calculate Fij
                        Fij[n,m] = (cosThetaI*cosThetaJ*AiMag*AjMag)/(pow(rMag,2)*np.pi)/AiMag
                        Fijsum[n] = Fijsum[n] + Fij[n,m]

        #Sky view factors
        for n in range(len(i)):
            AiMag = np.linalg.norm(Ai[n,:])

            for m in range(len(jSky)):
                r = i[n,:]-jSky[m,:] #distance vector between the two positions i[n] and j[m]
                rMag = np.linalg.norm(r)

                if ((float(rMag) > 0)):
                    if (bool_is_intersectedSky[n][m]):
                        continue   #skip the remaining part if ray intersects walls

                    AjMag = np.linalg.norm(AjSky[m,:])

                    ni = Ai[n,:]/AiMag #unit vector for Ai
                    nj = AjSky[m,:]/AjMag #unit vector for Aj
                    cosThetaI = np.abs(np.dot(ni, r))/rMag
                    cosThetaJ = np.abs(np.dot(nj, r))/rMag

                    #test to make sure that the surfaces are looking towards each other
                    cosPhiTest = np.dot(Ai[n,:],i[n,:]-jSky[m,:]) #/ (np.linalg.norm(Ai[n,:])*np.linalg.norm(i[n,:]-j[m,:]))
                    if (float(cosPhiTest) < 0): #if the pedestrian surface is looking towards the other one
                        #calculate FijSky
                        FijSky[n,m] = (cosThetaI*cosThetaJ*AiMag*AjMag)/(pow(rMag,2)*np.pi)/AiMag
                        FijsumSky[n] = FijsumSky[n] + FijSky[n,m]

        #write view factors
        if not os.path.exists(path + '/UTCI/'):
            os.makedirs(path + '/UTCI/')

        if not os.path.exists(path + '/UTCI/pos/'):
            os.makedirs(path + '/UTCI/pos/')

        if not os.path.exists(path + '/UTCI/pos/' + str(P_i) + '/'):
            os.makedirs(path + '/UTCI/pos/' + str(P_i) + '/')

        np.savez_compressed(path + '/UTCI/pos/' + str(P_i) + '/F', a=Fij, b=Fijsum, c=FijSky, d=FijsumSky)

    else:
        loaded = np.load(path + '/UTCI/pos/' + str(P_i) + '/F.npz')
        Fij = loaded['a']
        Fijsum = loaded['b']
        FijSky = loaded['c']
        FijsumSky = loaded['d']

    return Fij, Fijsum, FijSky, FijsumSky

def calculate_Tumrt(timestep, path, i, Ai, Tambient_, cloudCover_, Idif_, Fij, Fijsum, FijSky, FijsumSky, t):
    data = np.loadtxt(path + '/postProcessing/surfaces/' + str(timestep) + '/qrOut_wallAndTreeSurfaces.raw',skiprows=2)
    QrOut = data[:,3] #surface QrOut
    data = np.loadtxt(path + '/postProcessing/surfaces/' + str(timestep) + '/qsOut_wallAndTreeSurfaces.raw',skiprows=2)
    QsOut = data[:,3] #surface QsOut

    sigma = 5.67E-8 #W/m2/K4

    qin_lw = np.zeros([len(i),1]) #incident long-wave radiation to surface Ai
    qin_lw_wallAndTree = np.zeros([len(i),1])
    qin_sw = np.zeros([len(i),1]) #incident short-wave radiation to surface Ai
    qin_sw_wallAndTree = np.zeros([len(i),1])
    Eps_lw_person = 0.97
    Abs_sw_person = 0.7

    for n in range(len(i)):
        qin_lw_wallAndTree[n] = np.sum( np.multiply(QrOut, Fij[n,:]) )
        qin_sw_wallAndTree[n] = np.sum( np.multiply(QsOut, Fij[n,:]) )

    print('qin_lw_wallAndTree: ', str(qin_lw_wallAndTree.tolist()))
    print('qin_sw_wallAndTree: ', str(qin_sw_wallAndTree.tolist()))

    if t == 0:
        print('Fijsum: ' + str(Fijsum.tolist()) )
        print('FijsumSky: ' + str(FijsumSky.tolist()) )

    #sky
    cc = cloudCover_[t] #cloud cover
    ec = (1-0.84*cc)*(0.527 + 0.161*np.exp(8.45*(1-273/Tambient_[t]))) +0.84*cc #cloud emissivity
    Tsky = pow(9.365574E-6*(1-cc)*pow(Tambient_[t],6) + pow(Tambient_[t],4)*cc*ec ,0.25) #Swinbank model (1963, Cole 1976)
    qin_lw_sky = (sigma*pow(Tsky,4))*FijsumSky
    qin_sw_sky = Idif_[t]*FijsumSky
    print('qin_lw_sky: ', str(qin_lw_sky.tolist()))
    print('qin_sw_sky: ', str(qin_sw_sky.tolist()))

    qin_lw = qin_lw_wallAndTree + qin_lw_sky
    qin_sw = qin_sw_wallAndTree + qin_sw_sky

    for Pside in range(len(Fijsum)):
        scale_factor = 1/(Fijsum[Pside]+FijsumSky[Pside])
        qin_lw[Pside] = qin_lw[Pside]*scale_factor #scale qin accordingly
        qin_sw[Pside] = qin_sw[Pside]*scale_factor #scale qin accordingly

    if t == 0:
        print('qin_lw: ' + str(qin_lw.tolist()) + '\nqin_sw: ' + str(qin_sw.tolist()))

    #calculate Tumrt
    Tumrt = pow((qin_lw*Eps_lw_person + qin_sw*Abs_sw_person)/(sigma*Eps_lw_person),0.25)
    AiNorm = np.linalg.norm(Ai, axis=1)
    TumrtAvg = sum(Tumrt*AiNorm.reshape(5,1))/sum(AiNorm)

    return TumrtAvg

def pedestrian_param(P_):
    #pedestrian parameters
    #create rectangular prism for pedestrian
    if (groundNormal[0] == 1) and (groundNormal[1] == 0) and (groundNormal[2] == 0):
    #x-dir is vertical
        i = np.array([ #person's position (side and top center points)
            [P_[0]-1.0, P_[1]-0.2, P_[2]],
            [P_[0]-1.0, P_[1]+0.2, P_[2]],
            [P_[0]-1.0, P_[1], P_[2]-0.2],
            [P_[0]-1.0, P_[1], P_[2]+0.2],
            [P_[0], P_[1], P_[2]]
            ],dtype=np.float32)
        Ai = np.array([ #person's area (side and top area vectors)
            [0,   -0.68, 0],
            [0,    0.68, 0],
            [0,    0,   -0.68],
            [0,    0,    0.68],
            [0.16, 0,    0]
            ],dtype=np.float32)
        return i, Ai
        
    elif (groundNormal[0] == 0) and (groundNormal[1] == 1) and (groundNormal[2] == 0):
    #y-dir is vertical
        i = np.array([ #person's position (side and top center points)
            [P_[0]-0.2, P_[1]-1.0, P_[2]],
            [P_[0]+0.2, P_[1]-1.0, P_[2]],
            [P_[0],     P_[1]-1.0, P_[2]-0.2],
            [P_[0],     P_[1]-1.0, P_[2]+0.2],
            [P_[0],     P_[1],     P_[2]]
            ],dtype=np.float32)
        Ai = np.array([ #person's area (side and top area vectors)
            [-0.68, 0,    0],
            [ 0.68, 0,    0],
            [ 0,    0,   -0.68],
            [ 0,    0,    0.68],
            [ 0,    0.16, 0]
            ],dtype=np.float32)
        return i, Ai
        
    elif (groundNormal[0] == 0) and (groundNormal[1] == 0) and (groundNormal[2] == 1):
    #z-dir is vertical
        i = np.array([ #person's position (side and top center points)
            [P_[0]-0.2, P_[1],     P_[2]-1.0],
            [P_[0]+0.2, P_[1],     P_[2]-1.0],
            [P_[0],     P_[1]-0.2, P_[2]-1.0],
            [P_[0],     P_[1]+0.2, P_[2]-1.0],
            [P_[0],     P_[1],     P_[2]]
            ],dtype=np.float32)
        Ai = np.array([ #person's area (side and top area vectors)
            [-0.68, 0,    0],
            [ 0.68, 0,    0],
            [ 0,   -0.68, 0],
            [ 0,    0.68, 0],
            [ 0,    0,    0.16]
            ],dtype=np.float32)
        return i, Ai

def read_files(path, fileName, timestep):
    if (fileName == 'Tambient'):
        s=open(path + '/0/air/' + fileName).read().replace('(',' ').replace(')',' ').replace(';','')
    elif (fileName == 'cloudCover'):
        fileExists = os.path.isfile(path + '/0/air/' + fileName)
        if(fileExists):
            s=open(path + '/0/air/' + fileName).read().replace('(',' ').replace(')',' ').replace(';','')
        else:
            print('No cloudCover file found. Assuming clear sky conditions...')
            var_interp = np.zeros(len(timestep))
            return var_interp
    elif (fileName == 'Idif'):
        s=open(path + '/constant/' + fileName).read().replace('(',' ').replace(')',' ').replace(';','')

    varFile = np.loadtxt(StringIO(s), usecols=(0,1))
    lo = -1
    hi = -1
    for id, row in enumerate(varFile):
        if (timestep[0] <= row[0] and lo<0):
            lo = id
        elif (timestep[-1] <= row[0] and hi<0):
            hi = id
    var = varFile[lo:hi+1,:]
    var_interp = np.interp(timestep,var[:,0],var[:,1])

    return var_interp

def is_intersected(i, end, endSky, fileName):

    # Read in stl file using vtk
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fileName)
    reader.MergingOn()
    reader.Update()
    mesh = reader.GetOutput()

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(mesh)
    obbTree.BuildLocator()
    
    startDict = {}
    startDictSky = {}
    for n in range(len(i)):
        startDict[n] = np.tile(i[n,:], (len(end),1))
        startDictSky[n] = np.tile(i[n,:], (len(endSky),1))
    allTests = {}
    allTestsSky = {}

    #ray between pedestrian and wall surfaces
    for dicti in range(len(startDict)):
        start = startDict[dicti]

        start_ = start+(end-start)*0.03
        end_ = start+(end-start)*0.97
        tests = np.zeros(len(start_), dtype=bool)
        
        for i, start_i in enumerate(start_):
            pointsVTKintersection = vtk.vtkPoints()
            rMag = np.linalg.norm(start_[i]-end_[i]) #distance vector between the two positions i[n] and j[m]
            if (rMag < rMagMax): #view factor for walls is not calculated for long distances
                code = obbTree.IntersectWithLine(start_[i], end_[i], pointsVTKintersection, None)

                pointsVTKIntersectionData = pointsVTKintersection.GetData()
                noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
                if (noPointsVTKIntersection > 0):
                    tests[i] = True #True if there is intersection with surface
        allTests[dicti] = tests
    
    #ray between pedestrian and sky surfaces
    for dicti in range(len(startDictSky)):
        start = startDictSky[dicti]

        start_ = start+(endSky-start)*0.03
        end_ = start+(endSky-start)*0.97
        tests = np.zeros(len(start_), dtype=bool)
        
        for i, start_i in enumerate(start_):
            pointsVTKintersection = vtk.vtkPoints()
            rMag = np.linalg.norm(start_[i]-end_[i]) #distance vector between the two positions i[n] and j[m]

            code = obbTree.IntersectWithLine(start_[i], end_[i], pointsVTKintersection, None)

            pointsVTKIntersectionData = pointsVTKintersection.GetData()
            noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
            if (noPointsVTKIntersection > 0):
                tests[i] = True #True if there is intersection with surface
        allTestsSky[dicti] = tests
        
    return allTests, allTestsSky

def checkImportedData(path, veg, j, jSky):
    if(veg == False):
        filePath = path + '/constant/air/polyMesh/boundary'
    elif(veg == True):
        filePath = path + '/constant/vegetation/polyMesh/boundary'

    totalFaces = 0

    with open(filePath, 'r') as f:
        for line in f.readlines():
            if 'nFaces' in line:
                line_splitted = line.replace('\t',' ').replace(';',' ').split()
                totalFaces += int(line_splitted[1])

    if(len(j)+len(jSky) != totalFaces):
        raise ValueError('Total number of faces does not match the mesh nFaces. Maybe not all patches are included in postProcess surfaces?')


    
