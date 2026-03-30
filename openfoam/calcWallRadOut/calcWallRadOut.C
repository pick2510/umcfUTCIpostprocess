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

namespace
{
    Foam::IOdictionary readFieldDictionary
    (
        const Foam::fvMesh& mesh,
        const Foam::Time& runTime,
        const Foam::word& fieldName
    )
    {
        const Foam::word oldTypeName = Foam::IOdictionary::typeName;
        const_cast<Foam::word&>(Foam::IOdictionary::typeName) = Foam::word::null;

        Foam::IOdictionary fieldDict
        (
            Foam::IOobject
            (
                fieldName,
                runTime.timeName(),
                mesh,
                Foam::IOobject::MUST_READ,
                Foam::IOobject::NO_WRITE,
                false
            )
        );

        const_cast<Foam::word&>(Foam::IOdictionary::typeName) = oldTypeName;
        const_cast<Foam::word&>(fieldDict.type()) = fieldDict.headerClassName();
        return fieldDict;
    }

    const Foam::dictionary& lookupBoundaryPatchDict
    (
        const Foam::dictionary& boundaryFieldDict,
        const Foam::word& patchName
    )
    {
        return boundaryFieldDict.lookupEntry(patchName, false, true).dict();
    }

    Foam::scalarField readLookupScalarField
    (
        const Foam::dictionary& boundaryFieldDict,
        const Foam::word& patchName,
        const char* key
      , const Foam::label patchSize
    )
    {
        const Foam::dictionary& dict =
            lookupBoundaryPatchDict(boundaryFieldDict, patchName);

        if (!dict.found(key))
        {
            FatalErrorInFunction
                << "Missing key '" << key << "' for patch " << patchName
                << Foam::exit(Foam::FatalError);
        }

        return Foam::scalarField(key, dict, patchSize);
    }
}

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
        const IOdictionary qrDict = readFieldDictionary(mesh, runTime, "qr");
        const IOdictionary qsDict = readFieldDictionary(mesh, runTime, "qs");
        const dictionary& qrBoundaryDict = qrDict.subDict("boundaryField");
        const dictionary& qsBoundaryDict = qsDict.subDict("boundaryField");
        
        forAll(qrBf, patchi)
        {
            if (isA<wallFvPatch>(mesh.boundary()[patchi]))
            {
                fvPatchScalarField& qrOuti = qrOutBf[patchi];
                fvPatchScalarField& qsOuti = qsOutBf[patchi];
                const scalar sigma = constant::physicoChemical::sigma.value();

                fvPatchScalarField& qrPatch = qrBf[patchi];
                fvPatchScalarField& qsPatch = qsBf[patchi];

                if
                (
                    !isA<Foam::greyDiffusiveViewFactorFixedValueFvPatchScalarField>(qrPatch)
                 || !isA<Foam::solarLoad::solarLoadViewFactorFixedValueFvPatchScalarField>(qsPatch)
                )
                {
                    Info<< "Skipping patch " << mesh.boundary()[patchi].name()
                        << " because qr/qs patch field types are "
                        << qrPatch.type() << " / " << qsPatch.type()
                        << " instead of greyDiffusiveRadiationViewFactor / "
                        << "solarLoadRadiationViewFactor" << endl;
                    continue;
                }

                Foam::greyDiffusiveViewFactorFixedValueFvPatchScalarField& qrp =
                    refCast
                    <
                        Foam::greyDiffusiveViewFactorFixedValueFvPatchScalarField
                    >(qrPatch);
                Foam::solarLoad::solarLoadViewFactorFixedValueFvPatchScalarField& qsp =
                    refCast
                    <
                        Foam::solarLoad::solarLoadViewFactorFixedValueFvPatchScalarField
                    >(qsPatch);

                const fvPatchScalarField& Ti = TBf[patchi];
                const fvPatchScalarField& qri = qrBf[patchi];
                const fvPatchScalarField& qsi = qsBf[patchi];

                const word& patchName = mesh.boundary()[patchi].name();
                const dictionary& qrPatchDict =
                    lookupBoundaryPatchDict(qrBoundaryDict, patchName);
                const dictionary& qsPatchDict =
                    lookupBoundaryPatchDict(qsBoundaryDict, patchName);

                scalarField E;
                if
                (
                    qrPatchDict.lookupOrDefault<word>("emissivityMode", "")
                    == "lookup"
                )
                {
                    E = readLookupScalarField
                    (
                        qrBoundaryDict,
                        patchName,
                        "emissivity",
                        qrPatch.size()
                    );
                }
                else
                {
                    E = qrp.emissivity();
                }

                scalarField A;
                if (qsPatchDict.lookupOrDefault<word>("albedoMode", "") == "lookup")
                {
                    A = readLookupScalarField
                    (
                        qsBoundaryDict,
                        patchName,
                        "albedo",
                        qsPatch.size()
                    );
                }
                else
                {
                    A = qsp.albedo();
                }
                
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
