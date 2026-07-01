#include <viam/trajex/totg/tcp_jacobian.hpp>

#include <memory>

#include <viam/trajex/jacobian/jacobian.hpp>

namespace viam::trajex::totg {

tcp_jacobian make_tcp_jacobian(const xt::xarray<double>& model_table) {
    auto chain = std::make_shared<jacobian::kinematic_chain>(jacobian::kinematic_chain::from(model_table));
    return tcp_jacobian{
        .jacobian = [chain](const xt::xarray<double>& q) -> xt::xarray<double> { return chain->linear_jacobian(q); },
        .velocity_derivative = [chain](const xt::xarray<double>& q,
                                       const xt::xarray<double>& q_prime,
                                       const xt::xarray<double>& q_double_prime) -> trajectory::tcp_limit::gain_derivative {
            const auto vg = chain->velocity_gain_and_derivative(q, q_prime, q_double_prime);
            return {vg.gain, vg.dgain_ds};
        },
    };
}

}  // namespace viam::trajex::totg
