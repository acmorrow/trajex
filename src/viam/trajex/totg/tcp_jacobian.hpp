#pragma once

#include <functional>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/trajectory.hpp>

namespace viam::trajex::totg {

/// Bundles the two callbacks a trajectory::tcp_limit needs: the Jacobian (limit value) and the
/// gain derivative (limit slope), built once from one (n, 10) model-table tensor.
struct tcp_jacobian {
    std::function<xt::xarray<double>(const xt::xarray<double>&)> jacobian;
    std::function<trajectory::tcp_limit::gain_derivative(const xt::xarray<double>&, const xt::xarray<double>&, const xt::xarray<double>&)>
        velocity_derivative;
};

/// Builds the TCP Jacobian callbacks from an (n, 10) model-table tensor (viam::sdk::ModelTable
/// format). The tensor is parsed once into a jacobian::kinematic_chain shared by both callbacks.
///
/// @param model_table (n, 10) tensor in the viam::sdk::ModelTable format
/// @return jacobian (q -> 3xN linear Jacobian) and velocity_derivative ((q, q', q'') -> {gain, dgain_ds})
/// @throws std::invalid_argument on a malformed model-table tensor (see jacobian::kinematic_chain::from)
[[nodiscard]] tcp_jacobian make_tcp_jacobian(const xt::xarray<double>& model_table);

}  // namespace viam::trajex::totg
