/*---------------------------------------------------------------------------* \
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    calculateqrsw

Description
    calculateqrsw by cnevers

\*---------------------------------------------------------------------------*/


#include "argList.H"
#include "fvMesh.H"
#include "Time.H"
#include "fvc.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "distributedTriSurfaceMesh.H"
#include "triSurfaceTools.H"
#include "OFstream.H"
#include "meshTools.H"
#include "meshSearch.H"
#include "plane.H"
#include "uindirectPrimitivePatch.H"
#include "DynamicField.H"
#include "IFstream.H"
#include "unitConversion.H"

#include "mathematicalConstants.H"
#include "scalarMatrices.H"
#include "CompactListList.H"
#include "labelIOList.H"
#include "labelListIOList.H"
#include "scalarListIOList.H"
#include "vectorListIOList.H"
#include "scalarIOList.H"
#include "vectorIOList.H"

#include "singleCellFvMesh.H"
#include "interpolation.H"
#include "IOdictionary.H"
#include "fixedValueFvPatchFields.H"
#include "wallFvPatch.H"
#include "uniformDimensionedFields.H"
#include "unitConversion.H"
#include "timeSelector.H"
#include "Tuple2.H"

using namespace Foam;

// calculate the end point for a ray hit check
point calcEndPoint
(
    const point &start,
    const point &n2,
    const point &pminO,
    const point &pmaxO
)
{
  scalar ix = 0; scalar iy = 0; scalar iz = 0;

  if (n2.x() > 0.0)
    ix = (pmaxO.x() - start.x())/n2.x();
  else if (n2.x() < 0.0)
    ix = (pminO.x() - start.x())/n2.x();
  else
    ix = VGREAT;

  if (n2.y() > 0.0)
    iy = (pmaxO.y() - start.y())/n2.y();
  else if (n2.y() < 0.0)
    iy = (pminO.y() - start.y())/n2.y();
  else iy = VGREAT;

  if (n2.z() > 0.0)
    iz = (pmaxO.z() - start.z())/n2.z();
  else if (n2.z() < 0.0)
    iz = (pminO.z() - start.z())/n2.z();
  else
    iz = VGREAT;

  // closest edg direction
  scalar i = min(ix, min(iy, iz));

  return 0.9999*i*n2 + start;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
void readTimeTable
(
    const fileName& path,
    scalarField& times,
    Field<Type>& values
)
{
    IFstream is(path);
    if (!is.good())
    {
        FatalErrorInFunction
            << "Cannot read table " << path
            << exit(FatalError);
    }

    List<Tuple2<scalar, Type>> table(is);
    times.setSize(table.size());
    values.setSize(table.size());
    forAll(table, i)
    {
        times[i] = table[i].first();
        values[i] = table[i].second();
    }
}

void createRegionMesh
(
    const argList& args,
    Time& runTime,
    autoPtr<fvMesh>& meshPtr
)
{
    const word regionName =
        args.optionLookupOrDefault<word>("region", fvMesh::defaultRegion);
    const word timeName = Time::timeName(runTime.value());
    Info<< "Create mesh " << regionName << " for time = "
        << timeName << nl << endl;
    meshPtr.reset
    (
        new fvMesh
        (
            IOobject
            (
                regionName,
                timeName,
                runTime,
                IOobject::MUST_READ
            )
        )
    );
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // start timer
    clock_t tstart = std::clock();
    clock_t tstartStep;

    timeSelector::addOptions();
    #include "addRegionOption.H"
    #include "setRootCase.H"
    #include "createTime.H"

    instantList timeDirs = timeSelector::select0(runTime, args);

    runTime.setTime(timeDirs.last(), timeDirs.size()-1);

    autoPtr<fvMesh> meshPtr;
    createRegionMesh(args, runTime, meshPtr);
    fvMesh& mesh = meshPtr();

    volScalarField qr
    (
      IOobject
      (
          "qr",
          Time::timeName(runTime.value()),
          mesh,
          IOobject::MUST_READ,
          IOobject::NO_WRITE
      ),
      mesh
    );

    //wordList boundaryTypes = Qr.boundaryField().types();
    wordList boundaryTypes(qr.boundaryField().types().size(), "zeroGradient");

    //Info << "boundaryTypes = " << boundaryTypes << endl;

    // Read sunPosVector list
    // interpolationTable<vector> sunPosVector
    // (
    //     runTime.time().rootPath()
    //     /runTime.time().globalCaseName()
    //     /runTime.time().constant()
    //     /"sunPosVector"
    // ); 

    scalarField sunPosVector_x;
    vectorField sunPosVector_y;
    readTimeTable
    (
        runTime.time().rootPath()
       /runTime.time().globalCaseName()
       /runTime.time().constant()
       /"sunPosVector",
        sunPosVector_x,
        sunPosVector_y
    ); 

    // interpolationTable<scalar> IDN // direct solar radiation intensity flux
    // (
    //     runTime.time().rootPath()
    //     /runTime.time().globalCaseName()
    //     /runTime.time().constant()
    //     /"IDN"
    // );

    scalarField IDN_x;
    scalarField IDN_y;
    readTimeTable
    (
        runTime.time().rootPath()
       /runTime.time().globalCaseName()
       /runTime.time().constant()
       /"IDN",
        IDN_x,
        IDN_y
    );


    #include "readGravitationalAcceleration.H"
    Info << "Gravity is = " << g << endl;

    const vector ez = - g.value()/mag(g.value());
    Info << "Vertical vector : " << ez << endl;

    // Mesh setup
    int nMeshCells = mesh.cells().size();

    // Mesh cell centers
    pointField pmeshC = mesh.C();

    // mesh bounding box
    point pminO = gMin(pmeshC);
    point pmaxO = gMax(pmeshC);

    // Set up searching engine for obstacles (copied from Aytac)
    #include "searchingEngine.H"

    tstartStep = std::clock();

    forAll(timeDirs, timeI)
    {

        runTime.setTime(timeDirs[timeI], timeI);
        const word timeName = Time::timeName(runTime.value());
        Info << nl << "Time = " << timeName << endl;

        // start clock
        tstartStep = std::clock();

        // Volume vector field qrsw 
        volVectorField qrswi
        (
            IOobject
            (
               "qrsw",
               timeName,
               mesh,
               IOobject::NO_READ,
               IOobject::AUTO_WRITE 
            ),
            mesh,
            dimensionedVector("0", dimensionSet(1,0,-2,0,0,0,0), vector::zero),
            boundaryTypes
        );
        
        // look for the correct range
        label lo = 0;
        label hi = 0;
        for (label i = 0; i < sunPosVector_y.size(); ++i)
        {
            if (runTime.value() >= sunPosVector_x[i])
            {
                lo = hi = i;
            }
            else
            {
                hi = i;
                break;
            }   
        }
        scalar hi_fraction = 0; 
        if (lo != hi) //if timestep is between two time values in sunPosVector
        {
            hi_fraction = (runTime.value() - sunPosVector_x[lo]) / (sunPosVector_x[hi] - sunPosVector_x[lo]);
        }
        
        // sunPosVector i
        vector n2 = sunPosVector_y[lo]*(1-hi_fraction) + sunPosVector_y[hi]*(hi_fraction);
        n2 /= mag(n2);

        // only if sun is above the horizon
        if ( (n2 & ez) > 0)
        {

            DynamicField<point> startListCells(nMeshCells);
            DynamicField<point> endListCells(nMeshCells);
            List<pointIndexHit> pHitListCells(nMeshCells);

            // Calculate solar short-wave radiation vector field
            forAll(qrswi, cellI)
            {
                point starti = pmeshC[cellI];
                point endi = calcEndPoint(starti, n2, pminO, pmaxO);
                startListCells.append(starti);
                endListCells.append(endi);
            }

            surfacesMesh.findLine(startListCells, endListCells, pHitListCells);

            forAll(qrswi, cellI)
            {
                if (!pHitListCells[cellI].hit())
                {
                    qrswi[cellI] = -n2*(IDN_y[lo]*(1-hi_fraction) + IDN_y[hi]*(hi_fraction));
                }
            }
        }

        // Export --- temp.
        //Info << "Info: Exporting step " << vectorID << endl;
        
        qrswi.correctBoundaryConditions();
        qrswi.write();

        runTime++;

        
        Info << "Solar ray direction " << lo*(1-hi_fraction)+hi*(hi_fraction)
             << ", It took " ;
        printf("%.3f", (std::clock()-tstartStep) / static_cast<double>(CLOCKS_PER_SEC));
        //printf("%.3f", (std::clock()-tstartStep) / (double)CLOCKS_PER_SEC);
        Info << " second(s)."<< endl;
        

    }

    // Status Info
    Info << "\nTotal time took: "
         //<< (std::clock()-tstart) / (double)CLOCKS_PER_SEC
         << (std::clock()-tstart) / static_cast<double>(CLOCKS_PER_SEC)
         <<" second(s).\n"<< endl;

    Info<< "End\n" << endl;
    return 0;

}


// ************************************************************************* //
