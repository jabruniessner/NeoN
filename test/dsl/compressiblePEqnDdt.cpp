// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Unit-level follow-up to neorhoPimpleFoam_investigation_summary.md §4.5 (NeoFOAM
// superproject, src/neofoam/solver/neorhoPimpleFoam.py): that section hand-derived that
// literal rhoPimpleFoam's `fvc::ddt(rho) + psi*correction(fvm::ddt(p))` reduces, within one
// not-yet-converged PISO pass, to exactly the same affine function of p_new as this bare
// `imp.ddt(psi, p)` -- and that the real compressible solver's pEqn additionally subtracts
// `exp.source(p_ddt_psi)` on top, which (per the same derivation) does NOT cancel against
// that; it adds `p_ddt_psi * V` to the RHS instead. Ablating that subtraction was tested
// empirically on the live solver and found to make it diverge (NEORHO_ZERO_PDDT_PSI in
// neorhoPimpleFoam.py, §4.5/§4.7) -- so it is a real, load-bearing, deliberately
// non-literal term, not a bug to remove. This test asserts both halves of that derivation
// directly against the assembled linear system, so the claim no longer rests on hand
// algebra or a full 200-step solver run.

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

namespace NeoN
{

// Mirrors ddtOperator.cpp's CreateVector helper: registering a field through a
// VectorCollection (rather than constructing it directly) is what gives it old-time
// storage, which oldTime(field) requires.
template<typename ValueType>
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    ValueType value;
    std::int64_t timeIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
        for (auto patchi : std::vector<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", value);
            bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
        }
        NeoN::Field<ValueType> domainVector(
            mesh.exec(),
            NeoN::Vector<ValueType>(mesh.exec(), mesh.nCells(), value),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", std::int64_t(0)},
             {"subCycleIndex", std::int64_t(0)},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};

TEST_CASE("compressible pEqn ddt(psi,p) matches the neorhoPimpleFoam §4.5 derivation")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    // psi_c = 3 (current/this-assembly value), psi_oldT = 1 (true previous-timestep value).
    fvcc::VolumeField<scalar>& psi = fieldCollection.registerVector<fvcc::VolumeField<scalar>>(
        CreateVector<scalar> {.name = "psi", .mesh = mesh, .value = 3.0, .timeIndex = 1}
    );
    fill(oldTime(psi).internalVector(), 1.0);

    // p_c = 4 (current/pre-solve value, also what p_ddt_psi below is built from),
    // p_oldT = 2 (true previous-timestep value).
    fvcc::VolumeField<scalar>& p = fieldCollection.registerVector<fvcc::VolumeField<scalar>>(
        CreateVector<scalar> {.name = "p", .mesh = mesh, .value = 4.0, .timeIndex = 1}
    );
    // CreateVector's domain vector only covers the nCells-sized internal portion; the
    // boundary slot stays zero-initialised until explicitly corrected from the BC dict
    // (matches ddtOperator.cpp's own CreateVector usage, which also calls this after
    // construction) -- needed here since the div(phiv,p) section below reads p's
    // boundary value, not just its old-time internal value.
    p.correctBoundaryConditions();
    fill(oldTime(p).internalVector(), 2.0);

    // p_ddt_psi = p_c * (psi_c - psi_oldT) / dt, the real solver's exact formula
    // (neorhoPimpleFoam.py: `ddt_psi = (psi - old_time(psi)) / dt; p_ddt_psi = p * ddt_psi`),
    // built directly from the hand-picked numbers above rather than via field arithmetic --
    // that arithmetic isn't what's in question here, only how exp::source(p_ddt_psi)
    // combines with imp::ddt(psi, p) inside one Expression is. Doesn't need old-time
    // storage, so a plain (unregistered) field is fine here.
    const scalar dt = 0.5;
    const scalar pDdtPsiValue = 4.0 * (3.0 - 1.0) / dt; // = 16
    auto pDdtPsiBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> pDdtPsi(
        exec, "p_ddt_psi", mesh, Vector<scalar>(exec, mesh.nCells(), pDdtPsiValue), pDdtPsiBCs
    );

    Dictionary ddtSchemes;
    ddtSchemes.insert("ddt(p)", std::string("BDF1"));
    Dictionary fvSchemes;
    fvSchemes.insert("ddtSchemes", ddtSchemes);

    SECTION("bare imp.ddt(psi, p) alone matches literal's ddt treatment " + execName)
    {
        // imp::ddt(...) alone returns a bare TemporalOperator, not an Expression -- it has
        // no .assemble(); use implicitOperation(ls, t, dt) directly, same as
        // ddtOperator.cpp's own "scalar-scalar ddt(rho, phi)" unit test.
        auto ddtOp = dsl::imp::ddt(psi, p);
        ddtOp.read(fvSchemes);
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);
        ddtOp.implicitOperation(ls, 1.0, dt);

        const auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

        for (auto ii = 0; ii < mtxValsV.size(); ++ii)
        {
            // diag = psi_c * V/dt = 3/0.5*V = 6V -- §4.5: "psi_c*V/dt*p_new", the coefficient
            // of p_new in both our bare ddt(psi,p) and literal's fvc::ddt(rho) +
            // psi*correction(fvm::ddt(p)) once rho_c=psi_c*p_c is substituted in.
            REQUIRE(mtxValsV[ii] == Catch::Approx(6.0 * volV[0]));
            // rhs = psi_oldT * p_oldT * V/dt = 1*2/0.5*V = 4V -- §4.5: "V/dt*rho_oldT" with
            // rho_oldT === psi_oldT*p_oldT by old-time consistency.
            REQUIRE(rhsV[ii] == Catch::Approx(4.0 * volV[0]));
        }
    }

    SECTION(
        "imp.ddt(psi,p) - exp.source(p_ddt_psi), the real pEqn term, adds exactly "
        "p_ddt_psi*V to the RHS and leaves the diagonal unchanged "
        + execName
    )
    {
        auto expr = dsl::imp::ddt(psi, p) - dsl::exp::source(pDdtPsi);
        expr.read(fvSchemes);
        auto ls = expr.assemble(mesh, 1.0, dt);

        const auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

        for (auto ii = 0; ii < mtxValsV.size(); ++ii)
        {
            // Diagonal is untouched by the explicit source: still psi_c*V/dt = 6V.
            REQUIRE(mtxValsV[ii] == Catch::Approx(6.0 * volV[0]));
            // §4.5: "ours = literal + extra term", extra term = -V/dt*p_c*(psi_c-psi_oldT) =
            // -p_ddt_psi*V (dt already folded into pDdtPsiValue) relative to the ddt-alone
            // case -- equivalently, in diag*p - rhs form: rhs_ours = rhs_alone + p_ddt_psi*V
            // = 4V + 16V = 20V. This is the extra, non-literal, empirically load-bearing
            // term (removing it diverges the solver -- see NEORHO_ZERO_PDDT_PSI in
            // neorhoPimpleFoam.py, §4.5/§4.7).
            REQUIRE(rhsV[ii] == Catch::Approx((4.0 + pDdtPsiValue) * volV[0]));
        }
    }

    SECTION(
        "(1/Cv)*exp.div(phiv,p) scales exactly like exp.div(phiv,p) alone, divided by Cv "
        + execName
    )
    {
        // EEqn's pressure-work (and K-ddt, K-div) terms all follow this exact pattern --
        // `(1.0 / gas.Cv) * nn.exp.div(phiv, p)` -- so this checks the one operation shared
        // by every explicit term in EEqn: does scalar-multiplying a SpatialOperator and then
        // assembling it give exactly the same result as assembling it unscaled and dividing
        // by hand? A bug here (wrong precedence, scaling the wrong operand, scaling only
        // part of the term) would silently scale every explicit term in EEqn incorrectly.
        // createSingleCellMesh's 4 boundary faces are symmetric (left/right and top/bottom
        // normals cancel in pairs) -- a uniform boundary value would make div(phiv,p)
        // vanish by construction regardless of any real scaling bug, silently passing. Use
        // distinct per-face values to guarantee a genuinely nonzero baseline.
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> phiv(exec, "phiv", mesh, surfaceBCs);
        phiv.boundaryData().value() = Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0});

        const scalar Cv = 10.0;

        Dictionary divSchemes;
        divSchemes.insert(
            "div(phiv,p)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
        );
        Dictionary divFvSchemes;
        divFvSchemes.insert("divSchemes", divSchemes);

        auto divOp1 = dsl::exp::div(phiv, p);
        divOp1.read(divFvSchemes);
        auto source1 = Vector<scalar>(exec, p.size(), zero<scalar>());
        divOp1.explicitOperation(source1);

        auto divOp2 = (1.0 / Cv) * dsl::exp::div(phiv, p);
        divOp2.read(divFvSchemes);
        auto source2 = Vector<scalar>(exec, p.size(), zero<scalar>());
        divOp2.explicitOperation(source2);

        const auto [hostSource1, hostSource2] = copyToHosts(source1, source2);
        const auto [vals1, vals2] = views(hostSource1, hostSource2);
        for (auto ii = 0; ii < vals1.size(); ++ii)
        {
            REQUIRE(vals2[ii] == Catch::Approx(vals1[ii] / Cv));
            // Also confirm the baseline itself is nonzero -- otherwise this test would pass
            // vacuously (0 == 0/Cv) without actually exercising the scaling.
            REQUIRE(vals1[ii] != Catch::Approx(0.0));
        }
    }
}

} // namespace NeoN
