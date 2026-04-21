/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2012 OpenFOAM Foundation
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
    calcSf

Description
    Calculates and writes mesh boundary area vector (Sf)
    
    - akubilay, July 2017
    - cnevers, August 2024

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "timeSelector.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
    #include "addRegionOption.H"
    #include "setRootCase.H"
    #include "createTime.H"
    instantList timeDirs = timeSelector::select0(runTime, args);
    const word regionName =
        args.optionLookupOrDefault<word>("region", fvMesh::defaultRegion);
    Info<< "Create mesh " << regionName << " for time = "
        << runTime.name() << nl << endl;
    fvMesh mesh
    (
        IOobject
        (
            regionName,
            runTime.name(),
            runTime,
            IOobject::MUST_READ
        )
    );

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info<< "Time = " << runTime.name() << endl;
        mesh.readUpdate();

        volVectorField Sf
        (
            IOobject
            (
                "Sf",
                runTime.name(),
                mesh
            ),
            mesh,
            dimensionedVector("Sf", dimensionSet(0,2,0,0,0,0,0), vector::zero)
        );

        volVectorField::Boundary& SfBf = Sf.boundaryFieldRef();

        forAll(SfBf, patchi)
        {
            SfBf[patchi] = mesh.Sf().boundaryField()[patchi];
        }

        Sf.write();
    }

    Info<< "End" << endl;

    return 0;
}

// ************************************************************************* //
