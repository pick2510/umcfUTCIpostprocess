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
    calcOutgoingRadiation

Description
    Calculates outgoing long-wave and short-wave radiation from boundaries
    
    - cnevers, August 2025

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "greyDiffusiveViewFactorFixedValueFvPatchScalarField.H"
#include "solarLoadViewFactorFixedValueFvPatchScalarField.H"
#include "wallFvPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
    #include "addRegionOption.H"
    #include "setRootCase.H"
    #include "createTime.H"
    instantList timeDirs = timeSelector::select0(runTime, args);
    #include "createNamedMesh.H"

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info<< "Time = " << runTime.timeName() << endl;
        mesh.readUpdate();

        volScalarField qrOut
        (
            IOobject
            (
                "qrOut",
                runTime.timeName(),
                mesh
            ),
            mesh,
            dimensionedScalar("qrOut", dimensionSet(1,0,-3,0,0,0,0), 0)
        );
        volScalarField::Boundary& qrOutBf = qrOut.boundaryFieldRef();

        volScalarField qsOut
        (
            IOobject
            (
                "qsOut",
                runTime.timeName(),
                mesh
            ),
            mesh,
            dimensionedScalar("qsOut", dimensionSet(1,0,-3,0,0,0,0), 0)
        );
        volScalarField::Boundary& qsOutBf = qsOut.boundaryFieldRef();

        volScalarField T
        (
            IOobject
            (
                "T",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        volScalarField qr
        (
            IOobject
            (
                "qr",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        volScalarField qs
        (
            IOobject
            (
                "qs",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        
        volScalarField::Boundary& TBf = T.boundaryFieldRef();
        volScalarField::Boundary& qrBf = qr.boundaryFieldRef();
        volScalarField::Boundary& qsBf = qs.boundaryFieldRef();
        
        forAll(qrBf, patchi)
        {
            if (isA<wallFvPatch>(mesh.boundary()[patchi]))
            {
                fvPatchScalarField& qrOuti = qrOutBf[patchi];
                fvPatchScalarField& qsOuti = qsOutBf[patchi];
                const scalar sigma = constant::physicoChemical::sigma.value();

                fvPatchScalarField& qrPatch = qrBf[patchi];
                Foam::greyDiffusiveViewFactorFixedValueFvPatchScalarField& qrp =
                    refCast
                    <
                        Foam::greyDiffusiveViewFactorFixedValueFvPatchScalarField
                    >(qrPatch);
                fvPatchScalarField& qsPatch = qsBf[patchi];
                Foam::solarLoad::solarLoadViewFactorFixedValueFvPatchScalarField& qsp =
                    refCast
                    <
                        Foam::solarLoad::solarLoadViewFactorFixedValueFvPatchScalarField
                    >(qsPatch);

                const fvPatchScalarField& Ti = TBf[patchi];
                const fvPatchScalarField& qri = qrBf[patchi];
                const scalarField E = qrp.emissivity();
                const fvPatchScalarField& qsi = qsBf[patchi];
                const scalarField A = qsp.albedo();
                
                qrOuti = sigma*pow4(Ti) + qri*(1-E)/E;
                qsOuti = scalar(0);
                forAll(qsOuti, faceI)
                {
                    const scalar denom = 1 - A[faceI];
                    if (denom > 1e-6)
                    {
                        qsOuti[faceI] = qsi[faceI]*A[faceI]/denom;
                    }
                }
            }
        }
        
        qrOut.write();
        qsOut.write();
        
    }

    Info<< "End" << endl;

    return 0;
}

// ************************************************************************* //
