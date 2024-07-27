#include <iostream>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>

/*! \mainpage

  This is a Revised Enskog Theory calculator for hard-sphere mixtures,
  implemented in C++ with a python interface.

  The C++ code is documented here, but the python interface shares
  identical class and member function names/arguments.

  An example use of the python interface is given in enskog.py, and
  repeated below:

  \code{.py}
import pyenskog
import math

#Set up the solver
kT = 1.0
verbose = False
#Tables are available up to 3rd order. However, a generalised N-order
#solution is also available and should be used instead.
use_tables = False
SonineOrder = 10
EOS = pyenskog.EOStype.BMCSL

#Here we compare with the 10th order results for an isotopic mixture
#presented by Erpenbeck in (1989):
density_close_pack = 2.0 / (math.sqrt(2))
density = density_close_pack / 3 # Volume of three times close packed
e = pyenskog.Enskog(density, EOS, SonineOrder, verbose, kT, use_tables)

#Add the fluid, equimolar isotopic
e.addSpecies(1.0, #diameter
             1.0, #mass
             0.5 #mol fraction
         )
e.addSpecies(1.0, #diameter
             0.1, #mass
             0.5 #mol fraction
         )

#Solve enskog theory
e.init()

#Access the calculation results, confirm the values
print "Value: calc = reference"
print "L11:", e.Lab(0,0),"=", +0.0021164
print "L21:", e.Lab(1,0),"=", -0.0021164
print "L12:", e.Lab(0,1),"=", -0.0021164
print "L22:", e.Lab(1,1),"=", +0.0021164

print "L1u:", e.Lau(0),"=", -0.0549928
print "L2u:", e.Lau(1),"=", +0.0549928
print "Lu1:", e.Lua(0),"=", -0.0549928
print "Lu2:", e.Lua(1),"=", +0.0549928

print "Luu:", e.Luu(),"=", +3.9197430

print "shear visc:", e.shearViscosity(),"=", +0.2937862
print "bulk visc:", e.bulkViscosity()
print "bulk/shear visc:", e.bulkViscosity() / e.shearViscosity()
  \endcode

  Hydrodynamic description
  ========================

  To eliminate ambiguity in the fluid description this library
  provides, the conservation equations and phenomological equations it
  represents are outlined here.

  We assume that the state of the system at some point in space can
  be completely described by the temperature \f$k_B\,T\f$, the
  velocity \f$\bm{v}\f$, and the number densities \f$\{n_a\}\f$ of
  each species.

  \f[
  \text{Variables:}\quad\left\{k_B\,T,\,\bm{v},\,\left\{n_a\right\}\right\}
  \f]

  The continuity equation for each species is

  ---

  \f{align*}{
  \frac{\partial n_a}{\partial t} = - \nabla\cdot\left[n_a\,\bm{v}_a\right]
  \f}

  ---
    
  Note that the total number density is \f$n=\sum_a n_a\f$, and the
  mass density is \f$\rho=\sum_a\rho_a=\sum_a m_a\,n_a\f$, where
  \f$m_a\f$ is the molecular mass of species \f$a\f$. The continuity
  equation above is used to describe the time evolution number
  densities. However, a number of additional mass balances arise
  from this expression, which are useful in simplifying later
  expressions. Multiplying the continuity equation by the particle
  mass \f$m_a\f$ gives:

  \f{align*}{
  \frac{\partial \rho_a}{\partial t} = - \nabla\cdot\left[\rho_a\,\bm{v}_a\right]
  \f}

  Summing for all species gives the overall mass continuity equation:

  \f{align*}{
  \frac{\partial \rho}{\partial t} &= - \nabla\cdot\left[\rho\,\bm{v}\right]
  \f}

  where we've defined the velocity as the mass-averaged velocity
  \f$\bm{v}=\rho^{-1}\sum_a\rho_a\,\bm{v}_a\f$. 

  We note that defining \f$\bm{v}\f$ as the mass-averaged velocity
  is a free choice. An alternative definition is the number averaged
  velocity; \f$\bar{\bm{v}}=n^{-1}\sum_a n_a\,v_a\f$. Summing the
  continuity equation for each species gives:

  \f{align*}{
  \frac{\partial n}{\partial t} = - \nabla\cdot\left[n\,\bar{\bm{v}}\right]
  \f}

  To calculate the velocities \f$\bm{v}_a\f$ from the mass-averaged
  velocity \f$\bm{v}\f$, we use the definition of the diffusive
  flux:
    
  ---

  \f{align*}{
  \bm{J}_a = \rho_a\left(\bm{v}_a - \bm{v}\right)
  \f}

  ---
    
  The diffusive flux is calculated using the phenomenological
  expression:

  ---

  \f{align*}{
  \bm{J}_a &= \sum_b L_{ab}\,\bm{X}_b + L_{au} \bm{X}_u
  & & \\
  \bm{X}_a &= - T\,\nabla \left(\frac{\mu_a}{T}\right)
  &
  \bm{X}_u &= - T^{-1}\,\nabla\,T
  \f}
    
  ---

  where \f$\mu_a\f$ is the chemical potential of species \f$a\f$
  (calculated using \ref Enskog::chemicalPotential), \f$L_{ab}\f$ is the
  mutual diffusion coefficient between species \f$a\f$ and \f$b\f$
  (calculated using \ref Enskog::Lab), and \f$L_{au}\f$ is the thermal
  diffusivity of species \f$a\f$ (calculated using \ref Enskog::Lau).

  The mass-averaged velocity is updated using the momentum
  continuity equation:
    
  ---

  \f{align*}{
  \rho\frac{\partial \bm{v}}{\partial t} = - \rho\,\bm{v}\cdot\nabla\bm{v} -\nabla\cdot\bm{P} 
  \f}

  ---

  where the pressure tensor is given by:

  ---

  \f{align*}{
  \bm{P} = p\,\bm{1} - \eta \left[\nabla\bm{v} + (\nabla\bm{v})^T\right] + \left((2/d)\eta - \eta_B\right)(\nabla\cdot\bm{v})\bm{1}
  \f}

  ---
    
  where \f$d=3\f$ is the dimensionality, \f$p\f$ is the isotropic
  pressure (calculated by \ref Enskog::pressure), \f$\eta\f$ is the shear
  viscosity (calculated by \ref Enskog::shearViscosity), and \f$\eta_B\f$ is
  the bulk viscosity (calculated by \ref Enskog::bulkViscosity).

  Finally, the time evolution of the temperature is given via the
  energy continuity equation:

  \f{align*}{
  \frac{\partial \rho\, e}{\partial t} = -\nabla \cdot \bm{J}_e
  \f}

  where \f$e=\bm{v}\cdot\bm{v}/2+u\f$ is the total specific
  energy. The energy flux, \f$\bm{J}_e\f$ is:

  \f{align*}{
  \bm{J}_e = \bm{J}_q + \rho\,e\,\bm{v} + \bm{P}\cdot\bm{v}
  \f}
    
  Where \f$\bm{J}_q\f$ is the diffusive energy flux, and is given by
  the phenomenological expression:

  ---

  \f{align*}{
  \bm{J}_q &= \sum_a L_{au}\,\bm{X}_a + L_{uu} \bm{X}_u
  \f}

  ---

  where \f$L_{uu}\f$ is the thermal conductivity (calculated by \ref
  Enskog::Luu).
    
  As the temperature is used to describe the state, we need to
  expand and rewrite the energy continuity equation to express it in
  terms of \f$k_B\,T\f$. Inserting \f$\bm{J}_e\f$ into the energy
  continuity equation:

  \f{align*}{
  \frac{\partial\rho\,e}{\partial t} &= -\nabla \cdot \left[\bm{J}_q + \rho\,e\,\bm{v} + \bm{P}\cdot\bm{v}\right]
  \\
  e\cancelto{0}{\left(\frac{\partial\rho}{\partial t}+\nabla\cdot\rho\,\bm{v}\right)}+\rho\frac{\partial e}{\partial t} &= -\nabla \cdot \left[\bm{J}_q + \bm{P}\cdot\bm{v}\right]  - \rho\,\bm{v} \cdot \nabla e
  \f}
    
  Where the term has cancelled as it is the mass continuity
  equation. Inserting the definition of \f$e\f$:
    
  \f{align*}{
  \bm{v} \cdot\left(\rho\,\frac{\partial \bm{v}}{\partial t} + \rho\,\bm{v}\cdot \nabla \bm{v}\right) + \rho\frac{\partial u}{\partial t} &= -\nabla \cdot \bm{J}_q -\nabla \cdot\bm{P}\cdot\bm{v}  - \rho\,\bm{v} \cdot \nabla\,u
  \f}
    
  Replacing the first term with the momentum continuity equation

  \f{align*}{
  \rho\frac{\partial u}{\partial t} &= -\nabla \cdot \bm{J}_q - \rho\,\bm{v} \cdot \nabla\,u + (\bm{v} \cdot \nabla\cdot\bm{P}-\nabla \cdot\bm{P}\cdot\bm{v})
  \f}
    
  Examining the term in parenthesis:
  \f{align*}{
  \bm{v} \cdot \nabla\cdot\bm{P}-\nabla \cdot\bm{P}\cdot\bm{v} &= v_i\,\nabla_j P_{ji}-\nabla_j\,P_{ji}\,v_i
  \\
  &= -P_{ji}\,\nabla_{j}\,v_i = -\bm{P}\mathbin{:}\nabla\bm{v}
  \f}

  Which results in the internal energy equation:

  \f{align*}{
  \rho\frac{\partial u}{\partial t} &= -\nabla \cdot \bm{J}_q - \rho\,\bm{v} \cdot \nabla\,u -\bm{P}\mathbin{:}\nabla\bm{v}
  \f}

  For hard spheres, \f$u=d\,n\,k_B\,T/(2\rho)\f$ is the specific
  internal energy. As we are working with number densities and
  temperature it is convenient to eliminate the mass density:
    
  \f{align*}{
  \rho\frac{\partial u}{\partial t} &= -\nabla \cdot \bm{J}_q - \rho\,\bm{v} \cdot \nabla\,u -\bm{P}\mathbin{:}\nabla\bm{v}
  \\
  \frac{\partial \rho\,u}{\partial t} - u\frac{\partial \rho}{\partial t} &= -\nabla \cdot \bm{J}_q - \bm{v} \cdot \nabla\rho\,\,u + u\bm{v} \cdot \nabla\rho -\bm{P}\mathbin{:}\nabla\bm{v}
  \\
  \frac{\partial \rho\,u}{\partial t} - u\cancelto{0}{\left(\frac{\partial \rho}{\partial t} + \nabla\cdot\rho\bm{v}\right)} &= -\nabla \cdot \bm{J}_q - \bm{v} \cdot \nabla\rho\,\,u - u\,\rho\nabla \cdot \bm{v} -\bm{P}\mathbin{:}\nabla\bm{v}
  \\
  \frac{\partial \rho\,u}{\partial t} &= -\nabla \cdot \bm{J}_q - \nabla\cdot\left[\rho\,\,u\,\bm{v}\right] -\bm{P}\mathbin{:}\nabla\bm{v}
  \f}

  Where the mass continuity equation has been eliminated
  again. Inserting the definition of \f$u\f$, the \f$\rho\f$ terms
  cancel:
    
  \f{align*}{
  \frac{3}{2}\frac{\partial n\,k_B\,T}{\partial t} &= -\nabla \cdot \bm{J}_q - \frac{3}{2} \nabla\cdot \left[n\,k_B\,T\,\bm{v}\right] - \bm{P}\mathbin{:}\nabla\bm{v}
  \f}

  We then extract the time dependence on the number density by
  using the chain rule:

  ---

  \f{align*}{
  \frac{3}{2}n\frac{\partial k_B\,T}{\partial t} + \frac{3}{2}k_B\,T\frac{\partial n}{\partial t} &= -\nabla \cdot \bm{J}_q - \frac{3}{2} \nabla\cdot \left[n\,k_B\,T\,\bm{v}\right] - \bm{P}\mathbin{:}\nabla\bm{v}
  \f}

  ---

  Simplified equations for fluid between two heated walls
  ==================================

  Assume we have two infinte rectangular parallel walls and a system
  which is symmetric in the directions perpendicular to the walls. The
  equations can be simplifed to:

  ---

  \f{align*}{
  \frac{\partial n_a}{\partial t} &= - \frac{\partial n_a \,v_{a,x}}{\partial x}
  \\
  \frac{\partial n}{\partial t} &= \sum_a \frac{\partial n_a}{\partial t}
  \\
  v_{a,x} &= \rho_a^{-1}\,J_{a_x} + v_{x}
  \\
  J_{a,x} &= - L_{au}\,T^{-1}\frac{\partial T}{\partial x} - \sum_b L_{ab}\,T\frac{\partial \mu_a\, T^{-1}}{\partial x}
  \\
  \frac{\partial v_x}{\partial t} &= -v_x\frac{\partial v_x}{\partial x} -\rho^{-1}\frac{\partial P_{xx}}{\partial x}
  \\
  P_{xx} &= p - \left(\frac{4}{3}\eta + \eta_B\right)\frac{\partial v_x}{\partial x}
  \\
  n\frac{\partial k_B\,T}{\partial t} &= k_B\,T\,\frac{\partial n}{\partial t} -\frac{2}{3}\frac{\partial J_{q,x}}{\partial x} - \frac{\partial n\,k_B\,T\,v_x}{\partial x} - \frac{2}{3}P_{xx}\frac{\partial v_x}{\partial x}
  \\
  J_{q,x} &= -\sum_a L_{au}\,T\frac{\partial \mu_a\, T^{-1}}{\partial x} - L_{uu}\,T^{-1}\frac{\partial T}{\partial x}
  \f}
  ---
*/

/*! \brief The class for performing the Enskog calculations.
*/
class Enskog {
public:
  /*! \brief Enumeration of the approximate equations of state.
   */
  typedef enum {
    BOLTZMANN, //!< Boltzmann/ideal gas, where \f$Z=1\f$ and \f$\chi_{ab}=1\f$.
    BMCSL, //!< The BMCSL equation of state.
    VS, //!< The VS equation of state.
    HEYES, //!< The Heyes Phys. Chem. Chem. Phys., 2019, 21, 6886 paper
  } EOStype;

  /*! \brief Constructor. 
    
    \param density The number density, \f$n\f$.
    \param eos The approximate equation of state to be used.
    \param sonineOrder The order of the Sonine approximation to be used.
    \param verbose If true, intermediate values of the Enskog
    calculation will be outputted for verification.
    \param kT The thermal unit \f$k_B\,T\f$.
    \param use_tables If true, tables will be used to calculate the Sonine integrals (only available up to the third Enskog approximation).
  */
  Enskog(double density,
	 int eos = BMCSL,
	 size_t sonineOrder = 3,
	 bool verbose = true,
	 double kT=1,
         bool use_tables = false):
    _density(density), _kT(kT), _eos(eos),
    _sonineOrder(sonineOrder), _verbose(verbose), _use_tables(use_tables)
  {}

  /*! \brief Calculate the compressibility factor, \f$Z\f$.

    As defined in Lopez de Haro et al (1983) Eq.(20b), and Xu and Stell (1989) Eq.(2.8).
    \f[
    Z = 1+\frac{2\,\pi\,n}{3}\sum_{a}^{S-1}\sum_{b}^{S-1} x_a\,x_b\,\chi_{ab}\,\sigma_{ab}^3
    \f]
  */
  double Z() const
  {
    double resZ = 0;
    for (size_t i(0); i < _species.size(); ++i)
      for (size_t j(0); j < _species.size(); ++j)
        resZ += _gr(i,j) *  std::pow(sigma(i,j), 3.0) * _species[i]._x * _species[j]._x;
        
    resZ *= _density * M_PI * 2.0 / 3.0;
    return resZ + 1;
  }

  /*! \brief Calculate the pressure \f$p=Z\,n\,k_B\,T\f$.*/
  double pressure() const
  { return Z() * _density * _kT; }

  /*! \brief Calculate the packing fraction. */
  double packing_fraction() const {

    double sum(0);
    for (const Species& s : _species)
      sum += s._x * std::pow(s._diameter, 3);
    sum *= M_PI / 6.0;
    
    return sum * _density;
  }
  
  void addSpecies(double diameter, double mass, double x) {
    _species.push_back(Species(diameter, mass, x));
  }

  /*! \brief Performs the calculations of the Sonine coefficients and
      radial distribution functions at contact. 
  */
  void init() {
    normalise_concs();

    const double packmax = std::sqrt(2) * M_PI / 6.0;
    if (packing_fraction() > packmax)
      throw std::runtime_error("Packing fraction (" + std::to_string(packing_fraction()) + ") is greater than the maximum for monocomponent spheres (" + std::to_string(packmax)+"), the EOS and kinetic theory are almost certainly wrong to use in this limit (which may not even be physical).");
    
    _gr = Eigen::MatrixXd();
    generate_gr_array();
    _a = Eigen::MatrixXd();
    _b = Eigen::MatrixXd();
    _d.clear();
    _h = Eigen::MatrixXd();
  }

  /*! \brief Total mass density \f$\rho\f$.

    \f[
    \rho = n\,\sum_a^s m_a\,x_a
    \f]
  */
  double rho() const
  {
    double sum(0);
    for (const Species& s : _species)
      sum += s._x * s._mass;

    return sum * _density;
  }

  /*! \brief Thermal diffusion coefficient \f$L_{au}\f$.

    As defined in Lopez de Haro et al (1983) Eq.(20b).
  */
  double Lau(size_t a)
  {
    if (!_a.size())
      Solvea();
    if (!_d.size())
      Solved();

    if (a >= _d.size())
      throw std::runtime_error("Lau(a,b) a is out of range!");
    
    double sum = 0.0;
    for (size_t b = 0; b < _species.size(); b++)
      sum += Kdd(b) * _species[b]._x * _density * _d[b](a);

    return 5.0 * _species[a]._x * _species[a]._mass * sum / (4.0 * _density) - _species[a]._x * _species[a]._mass * _a(a) / 2.0;
  }

  /*! \brief Thermal diffusion coefficient, \f$L_{ua}\f$. */
  double Lua (size_t a)
  {
    if (!_d.size())
      Solved();

    if (a >= _d.size())
      throw std::runtime_error("Lau(a,b) a is out of range!");

    double sum = 0.0;
    for (size_t b = 0; b < _species.size(); b++)
      sum += K(b) * _species[b]._x * _d[a](_species.size() + b) - Kdd(b) * _species[b]._x *_d[a](b);
    
    return - 5.0 * _species[a]._x * _species[a]._mass * sum / 4.0;
  }

  /*! \brief Mutual diffusion coefficient, \f$L_{ab}\f$. */
  double Lab(size_t a, size_t b) 
  {
    if (!_d.size())
      Solved();
    
    if (a >= _d.size() || (b >= _d.size()))
      throw std::runtime_error("Lab(a,b) a or b is out of range!");

      
    return _species[b]._x * _species[a]._x * _species[b]._mass * _species[a]._mass * _d[b](a) / 2.0;
  }

  /*! \brief Thermal conductivity, \f$L_{uu}\f$.
   */
  double Luu()
  {
    if (!_a.size())
      Solvea();
    if (!_d.size())
      Solved();
    double result = 0.0;

    double sum(0);
    for (size_t a = 0; a < _species.size(); a++)
      {
        double sum2(0);
        for (size_t b = 0; b < _species.size(); b++)
          sum2 += Kdd(b) * _species[b]._x * _d[b](_species.size() + a);
        sum += _species[a]._x * K(a) * (_a(_species.size() + a) - (5.0/2.0) * sum2);
      }
    
    result = 5.0 * sum / 4.0;

    double sum2 = 0.0;
    for (size_t a = 0; a < _species.size(); a++)
      for (size_t b = 0; b < _species.size(); b++)
	sum2 += std::sqrt(2.0 * M_PI * mu(a, b) * _kT) * std::pow(sigma(a,b),4) * _gr(a, b) * _species[a]._x * _species[b]._x * _density * _density / (_species[a]._mass+_species[b]._mass);
    
    result += sum2 * 4.0 / 3.0;

    sum = 0.0;
    for (size_t a = 0; a < _species.size(); a++)
      sum += Kdd(a) * Lau(a) / _species[a]._mass;

    result += sum * 5.0 / 2.0;
    return result * _kT;
  }

  /*! \brief Calculate the shear viscosity, \f$\eta\f$.
    
    As given by Erpenbeck (1989)
    
    \f{align*}{
    \eta = \frac{1}{2} \sum_{i=0}^{S-1} \frac{K_i'\,n_i\,k_B\,T}{n}b_0^{(i)} + \frac{4}{15}C
    \f}

    From Xu and Stell, we have an identical definition.

   */
  double shearViscosity()
  {
    if (!_b.size())
      Solveb();
    double sum = 0.0;
    for (size_t i = 0; i < _species.size(); ++i)
      sum += Kd(i) * _species[i]._x * _b(i);
    sum *= _kT / 2;

    return sum + (4.0 / 15.0) * C();
  }
  

  /*! \brief Bulk viscosity, \f$\eta_B\f$.

    As defined in Lopez de Haro et al (1984) Eq.(23), and Lopez de Haro
    (1983) Eq.(53).

    \f{align*}{
    \eta_B = \frac{4}{9}C+2\,k_B\,T\,\rho\sum_i\sum_j b_{ij}\,M_{ji}\,\chi_{ij}\,x_i\,h_1^{(i)}
    \f}
    

    Xu and Stell (1989), Give the following expression for the Bulk viscosity:
    \f{align*}{
    \eta_B = \frac{5}{2}k_B T\sum_{i=0}^{S-1} x_i(H_i-1)h^{(i)}_1 + \frac{4}{9} n^2 \sum_{i=0}^{S-1} \sum_{j=0}^{S-1} \sqrt{2 \pi k_B T \mu_{ij}} x_i x_j \sigma_{ij}^4 \chi_{ij}
    \f}
    
    Noting that they define:
    
    \f{align*}{
    H_i = 1+\frac{8\pi}{15} n \sum_{j=0}^{S-1} x_j M_{ji} \sigma_{ij}^3 \chi_{ij}
    \f}

    which is exactly the same definition as Lopez de Haro, therefore
    Xu and Stell have the same definition of the Sonine coefficients.
   */
  double bulkViscosity()
  {
    if (!_h.size())
      Solveh();
    double sum = 0.0;
    for (size_t i = 0; i < _species.size(); ++i)
      for (size_t j = 0; j < _species.size(); ++j)
	sum += _species[i]._x * bab(i,j) * M(j, i) * _gr(i,j) * _h(_species.size()+i);
    return (4.0 / 9.0) * C() + 2 * _kT * rho() * sum;
  }

  /*! \brief The radial distribution function at contact \f$\chi_{ij}\f$.
   */
  double gr(size_t i, size_t j) const {
    return _gr(i, j);
  }

  /*! \brief Calculate the chemical potential (per mass) of species
    \f$a\f$, \f$\mu_a\f$.

    Erpenbeck (1989) gives the chemical potential of an ideal gas
    (per mass) in Eq.~(23) as:

    \f[
    \mu_{a,ig} = \frac{3\,k_B\,T}{2\,m_a}\ln (2\,\pi\,\hbar^2) - \frac{3\,k_B\,T}{2\,m_a}\ln m_a - \frac{5\,k_B\,T}{2\,m_a} \ln (k_B\,T) + \frac{k_B\,T}{m_a} \ln (x_a\, p)
    \f]

    Grouping the constant terms, inserting \f$p=n\,k_B\,T\f$, and
    scaling out the mass and linear temperature dependence:

    \f{align*}{
    \frac{m_a\,\mu_{a,ig}}{k_B\,T} &= \frac{3}{2}\ln \left(\frac{2\,\pi\,\hbar^2}{m_a\,k_B\,T}\right) + \ln (x_a\,n)
    \\
    &= \frac{1}{2}\ln \left(\lambda^3\right) + \ln (x_a\,n)
    \f}
      
    where \f$\lambda=2\,\pi\,\hbar^2/(m_a\,k_B\,T)\f$. But this
    doesn't coincide with the definition in Erpenbeck (1989) Eq. (50),
    due to the missing factor of (\f$1/2\f$) on the first term. This
    is an error, and Erpenbeck should read $\lambda^{3/2}$. This can
    be verified by examining Eq. (51) where a factor of 1/2 reappears. 

    In Kincaid et al (1983), Eq.(15), the chemical potential for a
    BMCSL fluid is given (they call it CS, but comparison of
    Eq. (8a) confirms its BMCSL). In Erpenbeck (1989), the same
    expression is given in Eq. (71).
      
    The Kincaid expression is:
    \f{multline}{
    \frac{m_a\,\mu_{a,BMCSL}}{k_B\,T} = \ln(x_a\,n)
    - \ln(1-\zeta_3) + \frac{\pi\,\sigma_a^3\,p}{6\,k_B\,T}
    + \frac{3\,\zeta_2\,\sigma_a+3\,\zeta_1\sigma_a^2}{1-\zeta_3}
    \\
    +\frac{9\,\zeta_2^2\sigma_a^2}{2(1-\zeta_3)^2}
    + 3\left(\frac{\zeta_2\,\sigma_a}{\zeta_3}\right)^2\left(\ln (1-\zeta_3) + \frac{\zeta_3}{1-\zeta_3}-\frac{\zeta_3^2}{2(1-\zeta_3)^2}\right)
    \\
    -\left(\frac{\zeta_2\,\sigma_a}{\zeta_3}\right)^3\left(2\ln (1-\zeta_3) + \zeta_3 \frac{2-\zeta_3}{1-\zeta_3}\right) + C
    \f}

    Note: We have converted the Kincaid expression into per unit mass
    (to coincide with the Erpenbeck notation) and there is a typo in
    the Kincaid expression (they have an index 1 instead of (\f$i\f$ or
    \f$a\f$) on the 4th term on \f$\sigma\f$).

    Comparing with the corrected Erpenbeck (1989) Eq. (71) which again
    has the wrong power on \f$\lambda\f$, we have \f$C=\ln
    \lambda^{3/2}\f$, which is unusual as Kincaid state this is a
    constant, but this is not constant as it contains the temperature.
  */
  double chemicalPotential(size_t a) const
  {
    //A constant, which is ultimately irrelevant in the calculations
    const double hbar = 1.0;
    const double lambda = 2 * M_PI * hbar * hbar / (_species[a]._mass * _kT);

    const double siga = _species[a]._diameter;
    const double z1 = zeta(1);
    const double z2 = zeta(2);
    const double z3 = zeta(3);
    
    switch (_eos){
    case BOLTZMANN:
      return (_kT / _species[a]._mass) * ((3.0 / 2.0) * std::log(lambda) + std::log(_species[a]._x * _density));
    case BMCSL:
      return (_kT / _species[a]._mass) *
	(std::log(_species[a]._x * _density / (1 - z3))
	 + M_PI * std::pow(siga, 3) * pressure() / (6 * _kT)
	 + (3 * z2 * siga + 3 * z1 * std::pow(siga, 2)) / (1 - z3)
	 + 9 * std::pow(z2, 2) * std::pow(siga, 2) / (2 * std::pow(1 - z3, 2))
	 + 3 * std::pow(z2 * siga / z3, 2) * (std::log(1 - z3) + z3 / (1 - z3) - std::pow(z3, 2) / (2 * std::pow(1 - z3, 2)))
	 - std::pow(z2 * siga / z3, 3) * (2 * std::log(1 - z3) + z3 * (2 - z3) / (1 - z3))
	 + 3.0 * std::log(lambda));
    default:
      throw std::runtime_error("This EOS does not have chemical potentials implemented.");
    }
  }

  /*! \brief Sonine coefficient, \f$a_p^{(i)}\f$. */
  double c_a(size_t p, size_t i) const {
    return _a(p * _species.size() + i);
  }

  /*! \brief Sonine coefficient, \f$b_p^{(i)}\f$. */
  double c_b(size_t p, size_t i) const {
    return _b(p * _species.size() + i);
  }

  /*! \brief Sonine coefficient, \f$h_p^{(i)}\f$. */
  double c_h(size_t p, size_t i) const {
    return _h(p * _species.size() + i);
  }

  /*! \brief Sonine coefficient, \f$d_{i,r}^{(k)}\f$. 
    
    \f$i\f$ and \f$k\f$ are species, and \f$r\f$ is the sonine order.
   */
  double c_d(size_t i, size_t r, size_t k) const {
    return _d[k](r * _species.size() + i);
  }

  /*! \brief Performs validation calculations on the sonine coefficients.
    
    The tests are:

    \f{align*}{
    \sum_{i=0}^{S-1} \frac{\rho_i}{\rho} a_0^{i}&=0 
    \f}

    \f{align*}{
    \sum_{i=0}^{S-1} \frac{\rho_i}{\rho} d_{i,0}^{k}&=0 & \text{for }k\in[0,S-1]
    \f}

    \f{align*}{
    h_0^{i}&=0 & \text{for }i\in[0,S-1]
    \f}

    \f{align*}{
    \sum_{i=0}^{S-1} x_i h_1^{i}&=0 
    \f}
   */
  void validation() const {
    std::cout << "\\sum_{i=0}^{S-1} \\frac{\\rho_i}{\\rho} a_0^{i}&=0" << std::endl;
    {
      double sum(0);
      for (size_t i(0); i < _species.size(); ++i)
	sum += _species[i]._mass * _species[i]._x * c_a(0, i);
      std::cout << sum << "=0" << std::endl;
    }

    std::cout << "\\sum_{i=0}^{S-1} \\frac{\\rho_i}{\\rho} d_{i,0}^{k}&=0 & \\text{for }k\\in[0, S-1]" << std::endl;
    for (size_t k(0); k < _species.size(); ++k) {
      double sum(0);
      for (size_t i(0); i < _species.size(); ++i)
	sum += _species[i]._mass * _species[i]._x * c_d(i, 0, k);
      std::cout << sum << "=0" << std::endl;
    }

    std::cout << "h_0^{i}&=0 & \\text{for }i\\in[0, S-1]" << std::endl;
    for (size_t i(0); i < _species.size(); ++i)
      std::cout << c_h(0,i) << "=0" << std::endl;

    std::cout << "\\sum_{i=0}^{S-1} x_i h_1^{i}&=0" << std::endl;
    {
      double sum(0);
      for (size_t i(0); i < _species.size(); ++i)
	sum += _species[i]._x * c_h(1, i);
      std::cout << sum << "=0" << std::endl;
    }
  }

  /*! \brief Solve for the Sonine coefficients \f$h_{q}^{(i)}\f$.
    
    This function describes the basis of the Sonine calculation.
    
    All \f$S\times N\f$ Sonine coefficients (where \f$S\f$ is the
    number of species and \f$N\f$ is the degree of Sonine expansion)
    are encoded into one-dimensional arrays. For example, the
    \f$h_{q}^{(i)}\f$ values are encoded as \f$\{h_{q}^{(i)}\}=
    \{h_{(q\,S + i)}\} = \bm{h}\f$.
    
    To determine the Sonine coefficients, we need to solve a set of
    linear equations. The equations will be encoded into a matrix
    form:
    
    \f{align*}{
    \bm{A}\cdot\bm{h}=\bm{b}
    \f}
    
    \f$\bm{A}\f$ and \f$\bm{b}\f$ will vary for each set of
    coefficients. Once they are specified the solution is determined
    from the inversion of the matrix \f$\bm{A}\f$, i.e.
    \f$\bm{h}=\bm{A}^{-1}\cdot\bm{b}\f$.
    
    
    From Lopez de Haro et al (1983) Eq.(45), the linear equations to
    solve for the coefficients are:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=1}^{N-1}\Gamma_{ij}^{pq} h_{q}^{(j)} &= \frac{n_i^*}{n}\, K''_i\delta_{p1} & \text{for }i\in[0,\,S-1],\,p\in[1,\,N-1]
    \f}

    where 

    \f{multline*}{
    \Gamma_{ij}^{pq} = \delta_{ij} \sum_{l=0}^{S-1} \frac{n_i^*\,n_l^*}{n^2} \left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]'_{ij} 
    \\
    + \frac{n_i^*\,n_j^*}{n^2}\left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]''_{ij}
    \f}

    For each Sonine coefficient, the definition of \f$n_i^*\f$ and
    \f$\sigma_{ij}^{*2}\f$ differ. For \f$h_{q}^{(j)}\f$, we have
    \f$\sigma_{ij}^{*2}=\chi_{ij}\,\sigma_{ij}^2\f$ and
    \f$n_i^*=n_i\f$. Although this seems like a simple substitution
    the partial bracket integrals as defined by Lopez de Haro
    et. al. have a dependence on these parameters. To make this
    explicit we perform a substitution
    
    \f{align*}{
    \left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]'_{ij} &= \frac{\sigma_{ij}^{*2}}{\sigma_{ij}^2} \, {B^{pq}_{ij0}}'
    \\
    \left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]''_{ij} &= \frac{\sigma_{ij}^{*2}}{\sigma_{ij}^2} \, {B^{pq}_{ij0}}''
    \f}

    where \f${B^{pq}_{ij0}}'\f$ is the bracket integral evaluated as
    though \f$n_i^*=n_i\f$ and \f$\sigma_{ij}^{*2}=\sigma_{ij}^2\f$
    (i.e. for an ideal gas).

    We now write the expression in terms of these parameters:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=1}^{N-1}\left(\delta_{ij} \sum_{l=0}^{S-1} x_i\,x_l \chi_{il}{B_{ij0}^{pq}}'
    + x_i\,x_j\,\chi_{ij} {B_{ij0}^{pq}}'' \right) h_{q}^{(j)} &= x_i K''_i\delta_{p1} & \text{for }i\in[0,\,S-1],\,p\in[1,\,N-1]
    \f}

    Unlike the other sonine coefficient expressions, the sum begins at
    the first sonine polynomial \f$(p,q=1)\f$ rather than the zeroth
    \f$(p,q=0)\f$) as \f$h_{0}^{(i)}=0\f$ for all \f$i\f$. For
    symmetry with the solution of the other coefficients, here we
    extend the sum to include the zero-valued \f$q=0\f$ sonine
    polynomial coefficients:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}(1 - \delta_{q0})\left(\delta_{ij} \sum_{l=0}^{S-1} x_i\,x_l \chi_{il}{B_{ij0}^{pq}}'
    + x_i\,x_j\,\chi_{ij} {B_{ij0}^{pq}}'' \right) h_{q}^{(j)} &= x_i K''_i\delta_{p1} & \text{for }i\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    We need an additional set of equations to set
    \f$h_{0}^{(i)}=0\f$. Expressing this in in the same form as the
    previous equation:

    \f{align*}
    \sum_{j=0}^{S-1} \sum_{q=0}^{N-1} \delta_{q0} \delta_{ji} h_{q}^{(j)} &= 0 & \text{for }i\in[0,\,S-1]
    \f}

    We combining these linear equations by extending the range of
    \f$p\f$ to include \f$p=0\f$:

    \f{multline*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}\Bigg[(1 - \delta_{p0})(1 - \delta_{q0})\left(\delta_{ij} \sum_{l=0}^{S-1} x_i\,x_l \chi_{il}{B_{ij0}^{pq}}'
    + x_i\,x_j\,\chi_{ij} {B_{ij0}^{pq}}'' \right) 
    \\
    + \delta{p0}\,\delta_{q0}\,\delta_{ji}\Bigg]h_{q}^{(j)}
    = x_i K''_i\delta_{p1}
    \qquad\text{for }i\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}
    
    Xu and Stell (1989) note that these equations are not a set of
    independent equations as \f$\sum_{i=0}^{S-1} x_i\,K''_i = 0\f$. An
    additional equation required for a unique solution is:

    \f{align*}{
    \sum_{j=0}^{S-1} \,x_j\,h_1^{(j)} &= 0
    \f}

    To write this in a matrix form, we rewrite it as follows:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1} \delta_{q1}\,x_j\,h_q^{(j)} &= 0
    \f}

    In our implementation we replace the \f$p=1,i=0\f$ equation.

    To implement this, we have the following definitions of the
    matrices \f$\bm{A}\f$ and \f$\bm{b}\f$, which, in index notation
    are:
    
    \f{multline*}{
    A_{p\,S+i,q\,S+j} = (1-\delta_{p1}\delta_{i0})(1 - \delta_{p0})(1 - \delta_{q0}) \left(\delta_{ij} 
    \sum_{l=0}^{S-1} x_i\,x_l \chi_{il}{B_{ij0}^{pq}}'
    + x_i\,x_j\,\chi_{ij} {B_{ij0}^{pq}}'' \right) 
    \\
    + \delta{p0}\,\delta_{q0}\,\delta_{ji}
    + \delta_{p1}\,\delta_{i0}\,\delta_{q1}\,x_j
    \f}

    \f{align*}{
    b_{p\,S+i} = x_i K''_i\delta_{p1} (1-\delta_{p1}\delta_{i0})
    \f}

    This completes the statement of the solution for the
    \f$h_{q}^{(i)}\f$ coefficients!

    Comparison against Xu and Stell
    -------------------------------

    Xu and Stell (1989), give the following equation for the Sonine coefficients:
    
    \f{align*}{
    \sum_{q=1}^{N-1}\left[h_q^{(i)}\sum_{j=0}^{S-1} x_j {B^{pq}_{ij0}}'\chi_{ij} + \sum_{j=0}^{S-1} x_j {B^{pq}_{ij0}}''\chi_{ij} h_{q}^{(j)}\right] = T_i \delta_p1
    \f}

    where we have \f$T_i=K_{i}''\f$ Trying to frame their expression
    in terms of the Lopez de Haro et al form:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=1}^{N-1} \left[\delta_{ij}\sum_{l=0}^{S-1} x_i x_l {B^{pq}_{il0}}''\chi_{il} + x_i x_j {B^{pq}_{ij0}}''\chi_{ij}\right]h_q^{(j)} = x_i K''_i \delta_p1
    \f}

    Comparing this with the definition of \f$\Gamma_{ij}^{pq}\f$, we have:
    
    \f{align*}{
    \chi_{ij}\,{B^{pq}_{ij0}}'' &= \left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]'_{ij}
    \\
    \chi_{ij}\,{B^{pq''}_{ij0}}'' &= \left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]''_{ij}
    \f}

    Thus confirming the additional factor of \f$\chi_{ij}\f$ is
    required and the derivation above is correct.
   */
  void Solveh() {
    Eigen::MatrixXd A(_species.size() * _sonineOrder, _species.size() * _sonineOrder);
    Eigen::MatrixXd b(_species.size() * _sonineOrder, 1);

    for (size_t p = 1; p < _sonineOrder; p++)
      for (size_t i = 0; i < _species.size(); i++) {
        for (size_t q = 0; q < _sonineOrder; q++)
          for (size_t j = 0; j < _species.size(); j++)
	    if (q == 0)
              A(p * _species.size() + i, q * _species.size() + j) = 0;
	    else
	      {
		double sum = 0.0;
		if (i == j)
		  for (size_t l = 0; l < _species.size(); l++)
		    sum += _species[i]._x *_species[l]._x * Sd1o2(i,l,p,q) * _gr(i,l);
		sum += _species[i]._x * _species[j]._x * Sdd1o2(i,j,p,q)  * _gr(i,j);
		A(p * _species.size() + i, q * _species.size() + j) = sum;
	      }
	
	b(p * _species.size() + i) = (p == 1) ? (_species[i]._x * Kdd_deHaro(i)) : 0;
      }

    //Add the simple equations for h_0^{(i)}=0 (we use the p=0 slot)
    { const size_t p = 0;
      for (size_t i = 0; i < _species.size(); i++) {
        for (size_t q = 0; q < _sonineOrder; q++)
	  for (size_t j = 0; j < _species.size(); j++)
	    A(p * _species.size() + i, q * _species.size() + j) = (i == j) && (q == 0);

        b(p * _species.size() + i) = 0;	
      }
    }

    //Replace one (redundant) equation with the identity
    { const size_t p = 1;
      const size_t i = 0;
      for (size_t q = 0; q < _sonineOrder; q++)
	for (size_t j = 0; j < _species.size(); j++)
	  A(p * _species.size() + i, q * _species.size() + j)
	    = (q == 1) * _species[j]._x;
      
      b(p * _species.size() + i) = 0;
    }

    //Solve the equations
    _h = A.fullPivLu().solve(b);

    // const double relerror = (A * _h - b).norm() / b.norm();
    // if (relerror > 1e-6)
    //   std::cerr << "Warning! Solution for h did not converge!" << std::endl;
      
    // if (_verbose || (relerror > 1e-6)) {
    //   std::cout << "Solving for h" << std::endl;
    //   std::cout << "A\n" << A << std::endl;
    //   std::cout << "b\n" << b << std::endl;
    //   std::cout << "x\n" << _h << std::endl;
    //   std::cout << "Relative error = " << (A * _h - b).norm() / b.norm() << std::endl;
    // }
  }

  /*! \brief Simple structure to hold the species specific data
      (diameter, mass, fraction).
  */
  struct Species {
    Species(double d, double m, double x): _diameter(d), _mass(m), _x(x) {}
    double _diameter;
    double _mass;
    double _x;
  };

  std::vector<Species> _species;
  double _density;
  double _kT;
  int _eos;
  size_t _sonineOrder;
  bool _verbose;
  bool _use_tables;
  
  Eigen::MatrixXd _gr;
  Eigen::MatrixXd _a;
  Eigen::MatrixXd _b;
  Eigen::MatrixXd _h;
  std::vector<Eigen::MatrixXd> _d;

  /*! \brief Normalise the concentrations stored for each species to sum to one.*/
  void normalise_concs() {
    double sum(0);
    for (Species& s : _species)
      sum += s._x;
    
    for (Species& s : _species)
      s._x /= sum;
  }

  /*! \brief Interaction diameter \f$\sigma_{ij}=(\sigma_i+\sigma_j)/2\f$.
    
    As defined in Lopez de Haro et al (1983), after Eq.(2), and Erpenbeck
    (1989) Eq.~32.
  */
  double sigma(size_t i, size_t j) const
  { return (_species[i]._diameter + _species[j]._diameter) / 2; }

  /*! \brief The \f$C\f$ term of Erpenbeck.

    As defined in Lopez de Haro et al (1983), via comparison of
    Eq.(52), and Erpenbeck (1989) Eq.~(48).

    \f[
    C = \sum_a^s \sum_b^s n_a\, n_b\, \sigma_{ab}^4\,\chi_{ab} \sqrt{2\,\pi\,\mu_{ab}\,k_B\,T} 
    \f]
  */
  double C() const {
    double C(0);
    for (size_t i =0; i < _species.size(); i++)
      for (size_t j =0; j < _species.size(); j++)
        C += std::sqrt(2.0 * M_PI * mu(i, j) * _kT) * _species[i]._x * _density *_species[j]._x * _density * std::pow(sigma(i,j), 4.0) * _gr(i, j);
    return C;
  }

  /*! \brief Calculate (and cache) the radial distribution function at
    contact \f$\chi_{ab}\f$ for all pairs of species.*/
  void generate_gr_array() {
    //Only calculate the gr if not already calculated.
    if (_gr.size()) return;
    
    _gr.resize(_species.size(), _species.size());

    for (size_t i(0); i < _species.size(); ++i)
      for (size_t j(0); j < _species.size(); ++j)
        switch (_eos){
        case BOLTZMANN:
          _gr(i,j) = 1.0;
          break;
        case BMCSL:
          _gr(i,j) = (1.0 / (1.0 - zeta (3)))
            * (1.0 + ((3.0 * mud(i, j) * zeta (2)) / (1.0 - zeta (3))) +
               2.0 * pow ((zeta (2) * mud (i, j)) / (1.0 - zeta (3)), 2));
          break;
        case VS:
          _gr(i,j) = 1.0/(1-zeta(3)) 
            + (3.0-zeta(3) + 0.5 * zeta(3) * zeta(3)) * zeta(2) * mud(i,j) / std::pow(1.0 - zeta(3), 2)
            + (2.0 - zeta(3) - zeta(3) * zeta(3) * 0.5)
            * (2.0 * zeta(2) * zeta(2) + zeta(1) * zeta(3))
            * mud(i,j) * mud(i,j)  * 4.0 / (6.0 * std::pow(1 - zeta(3), 3));
          break;
        case HEYES:
	  {
	    if (_species.size() != 1)
	      throw std::runtime_error("EOS not implemented for multi-component systems.");

	    const double packfrac = packing_fraction();
	    const double x = packfrac / (1 - packfrac);
	    const double Z = 1 + 4 * x + 6 * std::pow(x, 2) + 2.3647684 * std::pow(x, 3) - 0.8698551 * std::pow(x, 4) + 1.1062803 * std::pow(x, 5) - 1.1014221 * std::pow(x, 6) + 0.66605866 * std::pow(x, 7) - 0.03633431 * std::pow(x, 8) - 0.20965164 * std::pow(x, 10) + 0.10555569 * std::pow(x, 14) - 0.00872380 * std::pow(x, 22);
	    
	    _gr(i,j) = (Z-1) * 3 / (2 * M_PI * _density * std::pow(_species[0]._diameter, 3));
	    break;
	  }
	default:
	  throw std::runtime_error("Not implemented.");
        }
  };

  /*! \brief The zeta function, \f$\zeta_i\f$, is a convenience
    function mainly used when calculating \f$\chi_{ab}\f$.

    \f[
    \zeta_i = \frac{\pi\,n}{6}\sum_a^s x_a\, \sigma_a^i 
    \f]
  */
  double zeta(size_t i) const
  {
    double sum(0);
    for (const Species& s : _species)
      sum += s._x * std::pow(s._diameter, i);    
    return sum * M_PI * _density / 6.0;
  }
  
  /*! \brief The mu (diameter) function is a convenience function
    mainly used when defining gr theories. */
  double mud(size_t i, size_t j) const
  {    
    return _species[i]._diameter * _species[j]._diameter / (_species[i]._diameter + _species[j]._diameter);
  }

  /*! \brief The effective mass \f$\mu_{ab}=\mu_a\,\mu_b/(\mu_a+\mu_b)\f$. */
  double mu(size_t a, size_t b) const
  {
    return _species[a]._mass * _species[b]._mass / (_species[a]._mass + _species[b]._mass);
  }

  ////////////////////////////////// Enskog Calculations///////////////////////////////////

  /*! \brief \f$M_{ab}=m_a / (m_a+m_b)\f$

    As defined in Lopez de Haro et al (1983), after Eq.(2), and Erpenbeck
    (1989) Eq.~48.
  */
  double M(size_t i, size_t j) const
  {
    return _species[i]._mass / (_species[i]._mass + _species[j]._mass);
  }
  
  /*! \brief The omega integral for hard spheres,
   \f$\left[\Omega^{(l,r)}_{ij}\right]_{hs}\f$.
   
   As defined in Lopez de Haro et. al. (1983), Eq.B2:
   
   \f{align*}{
   \left[\Omega^{(l,r)}_{ij}\right]_{hs} = \sqrt{\frac{2\,\pi\,k_B\,T}{\mu_{ij}}}\frac{(r+1)!}{4}\left[1-\frac{1+(-1)^l}{2(l+1)}\right]\sigma_{ij}^2
   \f}
   */
  double Oint (size_t i, size_t j, size_t l, size_t r) const
  {
    return std::sqrt(2.0 * M_PI * _kT / mu(i, j)) * std::tgamma(r+2) * 0.25 * (1 - (1.0 + std::pow(-1.0,l)) / (2 * (l + 1))) * std::pow(sigma(i,j), 2.0);
  }

  /*! \brief \f$b_{ab}\f$.

    As defined in Lopez de Haro et al (1983) Eq.(23b) and Erpenbeck
    (1989) Eq.~(48).
    \f[
    b_{ab} = \frac{2\,\pi\,n\, x_b\, \sigma_{ab}^3}{3\,\rho}
    \f]
  */
  double bab(size_t a, size_t b) const
  {
    return 2.0 * M_PI *_density * _species[b]._x * std::pow(sigma(a,b), 3) / (3.0 * rho());
  }
  
  /*! \brief \f$K_{i}\f$.

    As defined in Lopez de Haro et al (1983) Eq.(26a) and Erpenbeck
    (1989) Eq.~(48).
    \f[
    K_a = 1+\frac{12\,\rho}{5}\sum_b^s b_{ab}\,M_{ab}\,M_{ba}\,\chi_{ab}
    \f]
  */
  double K(size_t i) const
  {
    double sum = 0.0;
    for (size_t j = 0; j < _species.size(); ++j)
      sum += bab(i,j) * M(i,j) * M(j, i) * _gr(i, j);
    return 1.0 + 12.0 * rho() * sum / 5.0;
  }

  /*! \brief Calculate \f$K'_i\f$ (Lopez de Haro/Erpenbeck), which is
      equivalent to \f$H_i\f$ (Xu and Stell).

    As defined in Lopez de Haro et al (1983) Eq.(26b) and Erpenbeck
    (1989) Eq.~(48).

    \f[
    K'_i = 1+\frac{4\,\rho}{5}\sum_{j=0}^{S-1} b_{ij}\,M_{ji}\,\chi_{ij}
    \f]
    
    Xu and Stell give the following expression:

    \f{align*}{
    H_i &= 1+\frac{8\pi}{15} n \sum_{j=0}^{S-1} x_j M_{ji} \sigma_{ij}^3 \chi_{ij}
    \\
    &= 1+\frac{4\,\rho}{5} \sum_{j=0}^{S-1} \frac{2\,\pi\,n_j\sigma_{ij}^3}{3\,\rho} M_{ji} \chi_{ij}
    \\
    &= 1+\frac{4\,\rho}{5} \sum_{j=0}^{S-1} b_{ij} M_{ji} \chi_{ij}
    \f}
    
    Therefore we have confirmed that \f$K'_i=H_i\f$.
  */
  double Kd(size_t i) const
  {
    double sum(0);
    for (size_t j = 0; j < _species.size(); j++)
      sum += bab(i, j) * M(j, i) * _gr(i, j);
    return 1.0 + 4.0 * rho() * sum / 5.0;
  }

  /*! \brief Calculate \f$K''_{a}\f$ (Lopez de Haro et al), which is
      also \f$T_i\f$ (Xu and Stell).

    As defined in Lopez de Haro et al (1983) Eq.(26c).
    \f[
    K''_{a} = 1-Z+2\,\rho\sum_b^s b_{ab} \, M_{ba} \,\chi_{ab}
    \f]

    We can prove that \f$T_i=K_{i}''\f$, taking the definition of
    \f$T_i\f$ from Xu and Stell (1989) Eq.(2.7):

    \f{align*}{
    T_i &= 1 + \frac{5}{2}\frac{8\pi}{15} n \sum_{j=0}^{S-1} x_j M_{ji} \sigma_{ij}^3 \chi_{ij} - Z
    \\
    &= 1 + 2 \sum_{j=0}^{S-1} \rho \frac{2\pi n_j \sigma_{ij}^3}{3\rho}  M_{ji}  \chi_{ij} - Z = K_i''
    \f}
  */
  double Kdd_deHaro(size_t i) const
  {
    double sum(0);
    for (size_t j = 0; j < _species.size(); ++j)
      sum += bab(i,j) * M(j, i) * _gr(i, j);
    return 1.0 + 2.0 * rho() * sum  - Z();
  }

  /*! \brief \f$K''_{i}\f$ as defined in Erpenbeck (1989).

    As defined in Erpenbeck (1989) after Eq. (54).
    \f[
    K''_{a} = 1+\frac{4\,\rho}{5}\sum_b^s b_{ab} \, M_{ab} \,\chi_{ab}
    \f]

    NOTE, there is another definition! As used by Lopez de Haro, and
    available in \ref Enskog::Kdd_deHaro.
  */
  double Kdd(size_t i) const
  {
    double sum(0);
    for (size_t j = 0; j < _species.size(); ++j)
      sum += bab(i,j) * M(i, j) * _gr(i, j);
    return 1.0 + 4.0 * rho() * sum / 5.0;
  }
  
  /*! \brief Low-density partial bracket integral \f$\left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij}\f$.

    According to Lopez de Haro (1983), under Table 1, we have:

    \f[
    \left[F,\,G\right]'_{ij}=\left[G,\,F\right]'_{ij}
    \f]

    Which implies that this function is symmetric in p and q.

    These integrals implemented here are listed in Ferzig and
    Kaper. If \ref use_tables is false, then the general
    implementation will be used.
   */
  double Sd3o2 (size_t i, size_t j, size_t p, size_t q) const
  {
    if (!_use_tables)
      return Bpqdijl(i,j,p,q,1);

    //This function is symmetric in the p and q arguments
    if (q < p)
      return Sd3o2(i, j, q, p);

    if ((p == 0) && (q == 0))
      return 8.0*M(j,i)*Oint(i, j, 1, 1);
    
    if ((p == 0) && (q == 1))
      return 8.0*pow(M(j,i),2)*(2.5*Oint(i, j, 1, 1)-Oint(i, j, 1, 2));
    
    if ((p == 0) && (q == 2))
      return 4.0*pow(M(j,i),3)*((35.0/4.0)*Oint(i, j, 1, 1) - 7.0*Oint(i, j, 1, 2) + Oint(i, j, 1, 3));
    
    if ((p == 1) && (q == 1))
      return 8.0*M(j,i)*((5.0/4.0)*(6.0*pow(M(i,j),2)+5.0 * pow(M(j,i),2)) * Oint(i, j, 1, 1) - 5.0 * pow(M(j,i),2) * Oint(i, j, 1, 2) + pow(M(j,i),2) * Oint(i, j, 1, 3)+2.0*M(j,i)*M(i,j)*Oint(i, j, 2, 2));
    
    if ((p == 1) && (q == 2))
      return 8.0*pow(M(j,i),2)*((35.0/16.0)*(12.0*pow(M(i,j),2) + 5.0*pow(M(j,i),2))*Oint(i, j, 1, 1) - (21.0/8.0)*(4.0*pow(M(i,j),2) + 5.0*pow(M(j,i),2))*Oint(i, j, 1, 2) + (19.0/4.0)*pow(M(j,i),2)*Oint(i, j, 1, 3) - 0.5*pow(M(j,i),2)*Oint(i, j, 1, 4)+7.0*M(i,j)*M(j,i)*Oint(i, j, 2, 2) - 2.0*M(i,j)*M(j,i)*Oint(i, j, 2, 3));
    
    if ((p == 2) && (q == 2))
      return 8.0*M(j,i)*((35.0/64.0)*(40.0*pow(M(i,j),4)+168.0*pow(M(i,j)*M(j,i),2)+35.0*pow(M(j,i),4))*Oint(i, j, 1, 1)-(7.0/8.0)*pow(M(j,i),2)*(84.0*pow(M(i,j),2.0)+35.0*pow(M(j,i),2))*Oint(i, j, 1, 2) + (1.0/8.0)*pow(M(j,i),2)*(108.0*pow(M(i,j),2) + 133.0*pow(M(j,i),2))*Oint(i, j, 1, 3) - (7.0/2.0)*pow(M(j,i),4)*Oint(i, j, 1, 4) + (1.0/4.0)*pow(M(j,i),4)*Oint(i, j, 1, 5) + (7.0/2.0)*M(i,j)*M(j,i)*(4.0*pow(M(i,j),2)+7.0*pow(M(j,i),2))*Oint(i, j, 2,2) -14.0*M(i,j)*pow(M(j,i),3)*Oint(i, j, 2, 3) + 2.0*M(i,j)*pow(M(j,i),3)*Oint(i, j, 2, 4) + 2.0*pow(M(i,j),2)*pow(M(j,i),2)*Oint(i, j, 3, 3));

    std::cerr << "Got an sonine integral not implemented error q = " << q << " p = " << p << "\n";
    throw std::runtime_error("Sonine integral not implemented");
  }

  /*! \brief Low-density partial bracket integral \f$\left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]''_{ij}\f$.

    According to Lopez de Haro (1983), under Table 1, we have:

    \f[
    \left[F,\,G\right]''_{ij}=\left[G,\,F\right]''_{ji}
    \f]

    Which implies that a exchange of p and q requires i and j to be
    exchanged as well.

    These integrals implemented here are listed in Ferzig and
    Kaper. If \ref use_tables is false, then the general
    implementation will be used.
   */
  double Sdd3o2 (size_t i, size_t j, size_t p, size_t q) const
  {
    if (!_use_tables)
      return Bpqddijl(i,j,p,q,1);

    //Permute ij and qp at the same time
    if (q < p)
      return Sdd3o2(j, i, q, p);

    if ((p == 0) && (q == 0))
      return -8.0*sqrt(M(i,j)*M(j,i))*Oint(i, j, 1, 1);

    if ((p == 0) && (q == 1))
      return -8.0*sqrt(pow(M(i,j),3)*M(j,i))*((5.0/2.0)*Oint(i, j, 1,1) - Oint(i, j, 1,2));

    if ((p == 0) && (q == 2))
      return -4.0*sqrt(pow(M(i,j),5)*M(j,i))*((35.0/4.0)*Oint(i, j, 1, 1) - 7.0*Oint(i, j, 1, 2) + Oint(i, j, 1, 3));

    if ((p == 1) && (q == 1))
      return -8.0*sqrt(pow(M(i,j)*M(j,i),3))*((55.0/4.0)*Oint(i, j, 1,1)-5.0*Oint(i, j, 1,2) + Oint(i, j, 1, 3) - 2.0*Oint(i, j, 2, 2));

    if ((p == 1) && (q == 2))
      return -8.0*sqrt(pow(M(i,j),5)*pow(M(j,i),3))*((595.0/16.0)*Oint(i, j, 1,1)-(189.0/8.0)*Oint(i, j, 1,2)+(19.0/4.0)*Oint(i, j, 1, 3) -0.5*Oint(i, j, 1, 4) -7.0*Oint(i, j, 2,2) + 2.0*Oint(i, j, 2, 3));

    if ((p == 2) && (q == 2))
      return -8.0*sqrt(pow(M(i,j)*M(j,i),5))*((8505.0/64.0)*Oint(i, j, 1,1) - (833.0/8.0)*Oint(i, j, 1, 2) + (241.0/8.0)*Oint(i, j, 1,3) - (7.0/2.0)*Oint(i, j, 1,4) + 0.25*Oint(i, j, 1,5) - (77.0/2.0)*Oint(i, j, 2,2) + 14.0*Oint(i, j, 2,3) - 2.0*Oint(i, j, 2,4) + 2.0*Oint(i, j, 3, 3));

    std::cerr << "Got an sonine integral not implemented error q = " << q << " p = " << p << "\n";
    throw std::runtime_error("Sonine integral not implemented");
  }
  
  /*! \brief Low-density partial bracket integral \f$\left[S_{5/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right),\,S_{5/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right)\right]'_{ij}\f$.

    According to Lopez de Haro (1983), under Table 1, we have:

    \f[
    \left[F,\,G\right]'_{ij}=\left[G,\,F\right]'_{ij}
    \f]

    Which implies that this function is symmetric in p and q.

    These integrals implemented here are listed in Ferzig and
    Kaper. If \ref use_tables is false, then the general
    implementation will be used.
   */
  double Sd5o2 (size_t i, size_t j, size_t p, size_t q) const
  {
    if (!_use_tables)
      return Bpqdijl(i,j,p,q,2);

    //This function is symmetric in the p and q arguments
    if (q < p)
      return Sd5o2(i, j, q, p);

    if ((p == 0) && (q == 0))
      return (16.0/3.0)*M(j,i)*(5.0*M(i,j)*Oint(i, j, 1, 1) + (3.0/2.0)*M(j,i)*Oint(i, j, 2, 2));

    if ((p == 0) && (q == 1))
      return (16.0/3.0)*pow(M(j,i),2.0)*((35.0/2.0)*M(i,j)*Oint(i, j, 1, 1)-7.0*M(i,j)*Oint(i, j, 1, 2) + (21.0/4.0)*M(j,i)*Oint(i, j, 2,2) - (3.0/2.0)*M(j,i)*Oint(i, j, 2, 3));

    if ((p == 0) && (q == 2))
      return (16.0/3.0)*pow(M(j,i),3.0)*((315.0/8.0)*M(i,j)*Oint(i, j, 1, 1)-(63.0/2.0)*M(i,j)*Oint(i, j, 1, 2) + (9.0/2.0)*M(i,j)*Oint(i, j, 1, 3) + (189.0/16.0)*M(j,i)*Oint(i, j, 2,2) - (27.0/4.0)*M(j,i)*Oint(i, j, 2,3) + 0.75*M(j,i)*Oint(i, j, 2,4));

    if ((p == 1) && (q == 1))
      return (16.0/3.0)*M(j,i)*(0.25*M(i,j)*(140.0*pow(M(i,j),2.0)+245.0*pow(M(j,i),2.0))*Oint(i, j, 1, 1)-49.0*M(i,j)*pow(M(j,i),2.0)*Oint(i, j, 1, 2) + 8.0*M(i,j)*pow(M(j,i),2.0)*Oint(i, j, 1, 3)+(1.0/8.0)*M(j,i)*(154.0*pow(M(i,j),2.0) + 147.0*pow(M(j,i),2.0))*Oint(i, j, 2,2) - (21.0/2.0)*pow(M(j,i),3.0)*Oint(i, j, 2,3)+(3.0/2.0)*pow(M(j,i),3.0)*Oint(i, j, 2,4)+3.0*M(i,j)*pow(M(j,i),2.0)*Oint(i, j, 3,3));

    if ((p == 1) && (q == 2))
      return (16.0/3.0)*pow(M(j,i),2)*((1.0/16.0)*M(i,j)*(2520.0*pow(M(i,j),2) + 2205.0*pow(M(j,i),2))*Oint(i, j, 1, 1) - (1.0/8.0)*M(i,j)*(504.0*pow(M(i,j),2)+1323.0*pow(M(j,i),2))*Oint(i, j, 1, 2)+(207.0/4.0)*M(i,j)*pow(M(j,i),2)*Oint(i, j, 1, 3) - (9.0/2.0)*M(i,j)*pow(M(j,i),2)*Oint(i, j, 1, 4)+ (1.0/32.0)*M(j,i)*(2772.0*pow(M(i,j),2)+1323.0*pow(M(j,i),2))*Oint(i, j, 2, 2) - (1.0/16.0)*M(j,i)*(396.0*pow(M(i,j),2)+567.0*pow(M(j,i),2))*Oint(i, j, 2,3)+(75.0/8.0)*pow(M(j,i),3)*Oint(i, j, 2, 4) - (3.0/4.0)*pow(M(j,i),3)*Oint(i, j, 2, 5) + (27.0/2.0)*M(i,j)*pow(M(j,i),2)*Oint(i, j, 3,3) - 3.0*M(i,j)*pow(M(j,i),2)*Oint(i, j, 3, 4));
 
    if ((p == 2) && (q == 2))
      return 4.0*M(j,i)*((1.0/16.0)*M(i,j)*(2520.0*pow(M(i,j),4) + 15120.0*pow(M(i,j)*M(j,i),2) + 6615.0*pow(M(j,i),4))*Oint(i, j, 1, 1) -
			 0.5*M(i,j)*pow(M(j,i),2)*(1512.0*pow(M(i,j),2)+1323.0*pow(M(j,i),2))*Oint(i, j, 1, 2)+0.5*M(i,j)*M(j,i)*(252.0*pow(M(i,j),2) + 621.0*pow(M(j,i),2))*Oint(i, j, 1, 3) -
			 54.0*M(i,j)*pow(M(j,i),4)*Oint(i, j, 1, 4)+3.0*M(i,j)*pow(M(j,i),4)*Oint(i, j, 1, 5)+(1.0/32.0)*M(j,i)*(4536.0*pow(M(i,j),4)+16632.0*pow(M(i,j)*M(j,i),2)+3969.0*pow(M(j,i),4))
			 *Oint(i, j, 2, 2) - 0.25*pow(M(j,i),3)*(1188.0*pow(M(i,j),2)+567.0*pow(M(j,i),2))*Oint(i, j, 2,3)+0.25*pow(M(j,i),3)*(156.0*pow(M(i,j),2)+225.0*pow(M(j,i),2))*Oint(i, j, 2, 4) -
			 9.0*pow(M(j,i),5)*Oint(i, j, 2,5)+ 0.5*pow(M(j,i),5)*Oint(i, j, 2, 6)+M(i,j)*pow(M(j,i),2)*(42.0*pow(M(i,j),2)+81.0*pow(M(j,i),2))*Oint(i, j, 3, 3)- 36.0*M(i,j)*pow(M(j,i),4)*
			 Oint(i, j, 3, 4)+4.0*M(i,j)*pow(M(j,i),4)*Oint(i, j, 3, 5)+4.0*M(i,j)*pow(M(j,i),3)*Oint(i, j, 4, 4));

    std::cerr << "Got an sonine integral not implemented error q = " << q << " p = " << p << "\n";
    throw std::runtime_error("Sonine integral not implemented");
  }

  /*! \brief Low-density partial bracket integral \f$\left[S_{5/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right),\,S_{5/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right)\right]''_{ij}\f$.

    According to Lopez de Haro (1983), under Table 1, we have:

    \f[
    \left[F,\,G\right]''_{ij}=\left[G,\,F\right]''_{ji}
    \f]

    Which implies that a exchange of p and q requires i and j to be
    exchanged as well.

    These integrals implemented here are listed in Ferzig and
    Kaper. If \ref use_tables is false, then the general
    implementation will be used.
   */
  double Sdd5o2 (size_t i, size_t j, size_t p, size_t q) const
  {
    if (!_use_tables)
      return Bpqddijl(i,j,p,q,2);

    //Permute ij and qp at the same time
    if (q < p)
      return Sdd5o2(j, i, q, p);

    if ((p == 0) && (q == 0))
      return -(16.0/3.0)*M(i,j)*M(j,i)*(5.0*Oint(i, j, 1, 1) - (3.0/2.0)*Oint(i, j, 2, 2));

    if (((p == 1) && (q == 0)) || ((p == 0) && (q == 1)))
      return (16.0/3.0)*pow(M(i,j),2.0)*M(j,i)*((-35.0/2.0)*Oint(i, j, 1, 1)+7.0*Oint(i, j, 1, 2)+(21.0/4.0)*Oint(i, j, 2, 2)-(3.0/2.0)*Oint(i, j, 2,3));

    if (((p == 2) && (q == 0)) || ((p == 0) && (q == 2)))
      return (16.0/3.0)*pow(M(i,j),3)*M(j,i)*((-315.0/8.0)*Oint(i, j, 1, 1)+(63.0/2.0)*Oint(i, j, 1, 2) - (9.0/2.0)*Oint(i, j, 1, 3)+ (189.0/16.0)*Oint(i, j, 2, 2) - (27.0/4.0)*Oint(i, j, 2, 3) + 0.75*Oint(i, j, 2, 4));

    if ((p == 1) && (q == 1))
      return  -(16.0/3.0)*pow(M(j,i)*M(i,j),2.0)*((385.0/4.0)*Oint(i, j, 1, 1)-49.0*Oint(i, j, 1, 2)+8.0*Oint(i, j, 1, 3)-(301.0/8.0)*Oint(i, j, 2,2)+(21.0/2.0)*Oint(i, j, 2, 3)-(3.0/2.0)*Oint(i, j, 2, 4)+3.0*Oint(i, j, 3, 3));

  
    if (((p == 1) && (q == 2)) || ((p == 2) && (q == 1)))
      return (16.0/3.0)*pow(M(i,j),3)*pow(M(j,i),2)*((-4725.0/16.0)*Oint(i, j, 1, 1)+(1827.0/8.0)*Oint(i, j, 1, 2)- (207.0/4.0)*Oint(i, j, 1, 3) + (9.0/2.0)*Oint(i, j, 1, 4) + (4095.0/32.0)*Oint(i, j, 2, 2) - (963.0/16.0)*Oint(i, j, 2, 3) + (75.0/8.0)*Oint(i, j, 2, 4) - 0.75*Oint(i, j, 2, 5) - (27.0/2.0)*Oint(i, j, 3, 3)+3.0*Oint(i, j, 3, 4));


    if ((p == 2) && (q == 2))
      return 4.0*pow(M(i,j)*M(j,i),3)*((-24255.0/16.0)*Oint(i, j, 1, 1) + (2835.0/2.0)*Oint(i, j, 1, 2) -(873.0/2.0)*Oint(i, j, 1, 3)+54.0*Oint(i, j, 1, 4)-3.0*Oint(i, j, 1, 5)+(25137.0/32.0)*Oint(i, j, 2, 2) -(1755.0/4.0)*Oint(i, j, 2, 3)+ (381.0/4.0)*Oint(i, j, 2, 4) - 9.0*Oint(i, j, 2, 5)+0.5*Oint(i, j, 2, 6) - 123.0*Oint(i, j, 3, 3)+36.0*Oint(i, j, 3, 4)-4.0*Oint(i, j, 3, 5) + 4.0*Oint(i, j, 4, 4));

    std::cerr << "Got an sonine integral not implemented error q = " << q << " p = " << p << "\n";
    throw std::runtime_error("Sonine integral not implemented");
  }

  /*! \brief Low-density partial bracket integral \f$\left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]'_{ij}\f$.

    According to Lopez de Haro (1983), under Table 1, we have:

    \f[
    \left[F,\,G\right]'_{ij}=\left[G,\,F\right]'_{ij}
    \f]

    Which implies that this function is symmetric in p and q.

    These integrals implemented here are listed in Table 1 of Lopez de
    Haro (1983). If \ref use_tables is false, then the general
    implementation will be used.
   */
  double Sd1o2(size_t i, size_t j, size_t p, size_t q) const
  {
    if (!_use_tables)
      return Bpqdijl(i,j,p,q,0);
    
    //This function is symmetric in the p and q arguments
    if (q < p)
      return Sd1o2(i, j, q, p);
    
    if (p == 0) return 0.0;

    if ((p == 1) && (q == 1))
      return 16 * M(i,j) * M(j, i) * Oint(i, j, 1, 1);

    if ((p == 1) && (q == 2))
      return 16.0 * M(i,j) * std::pow(M(j, i), 2) * (5 * Oint(i, j, 1, 1) - 2 * Oint(i,j,1,2));

    if ((p == 2) && (q == 2))
      return 16 * std::pow(M(j, i), 3) * M(i,j) * (4 * Oint(i,j,1,3) - 24 * Oint(i,j,1,2) + 35 * Oint(i,j,1,1))
        + 64 * std::pow(M(i, j), 3) * M(j, i) * Oint(i,j,1,1) + 64 * std::pow(M(i, j), 2) * std::pow(M(j, i), 2) * Oint(i,j,2,2) + 32 * M(i, j) * M(j, i)
        * (M(j, i) - M(i, j)) * (2 * Oint(i,j,1,2) - 5 * Oint(i,j,1,1));
    
    std::cerr << "Got an sonine integral not implemented error q = " << q << " p = " << p << "\n";
    throw std::runtime_error("Sonine integral not implemented");
  }

  /*! \brief Low-density partial bracket integral \f$\left[S_{1/2}^{(p)}\left(m\,V^2/2\,k_B\,T\right),\,S_{1/2}^{(q)}\left(m\,V^2/2\,k_B\,T\right)\right]''_{ij}\f$.

    According to Lopez de Haro (1983), under Table 1, we have:

    \f[
    \left[F,\,G\right]''_{ij}=\left[G,\,F\right]''_{ji}
    \f]

    Which implies that a exchange of p and q requires i and j to be
    exchanged as well.

    These integrals implemented here are listed in Table 1 of Lopez de
    Haro (1983). If \ref use_tables is false, then the general
    implementation will be used.
   */
  double Sdd1o2(size_t i, size_t j, size_t p, size_t q) const
  {
    if (!_use_tables)
      return Bpqddijl(i,j,p,q,0);

    //Permute ij and qp at the same time
    if (q < p)
      return Sdd1o2(j, i, q, p);

    if (p == 0) return 0.0;

    if ((p == 1) && (q == 1))
      return -16 * M(i, j) * M(j, i) * Oint(i,j,1,1);

    if ((p == 1) && (q == 2)) 
      return -16 * std::pow(M(i, j), 2) * M(j, i) * (5 * Oint(i,j,1,1) - 2 * Oint(i,j,1,2));

    if ((p == 2) && (q == 2))
      return -16 * std::pow(M(i, j), 2) * std::pow(M(j, i), 2) * (4 * Oint(i,j,1,3) - 20 * Oint(i,j,1,2) + 35 * Oint(i,j,1,1) - 4 * Oint(i,j,2,2));

    std::cerr << "Got an sonine integral not implemented error q = " << q << " p = " << p << "\n";
    throw std::runtime_error("Sonine integral not implemented");
  }

  /*! \brief Solve for the Sonine coefficients \f$a_{q}^{(i)}\f$.
    
    Please see \ref Enskog::Solveh for more information on the decoding of the
    notation of Lopez de Haro, and how the solution is carried out.
    
    The linear equations to solve for the coefficients are (from Lopez
    de Haro et al (1983) Eq.(43)):

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}\Lambda_{ij}^{pq}\,a_{q}^{(j)} &= \frac{4}{5\,k_B} \frac{n_i^*}{n_i}\delta_{p1} & \text{for }i\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    where
    
    \f{multline*}{\scriptsize
    \Lambda_{ij}^{pq} = \frac{8\sqrt{m_i\,m_j}}{75\,k_B^2\,T}\left(\delta_{ij}\sum_{l=0}^{S-1}\frac{n_i^*\,n_l^*}{n^2} \left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij}
    +\frac{n_i^*\,n_j^*}{n^2}\left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]''_{ij}\right)
    \f}

    and \f$n_i^* = n_i\,K_i\f$ and
    \f$\sigma_{ij}^{*2}=\chi_{ij}(K_i\,K_j)^{-1}\sigma_{ij}^2\f$.

    Taking out the hidden dependence of the partial bracket integrals, we have:
    
    \f{align*}
    \left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij}  &= \chi_{ij}(K_i\,K_j)^{-1}{B_{ij1}^{pq}}'
    \\
    \left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij} &= \chi_{ij}(K_i\,K_j)^{-1}{B_{ij1}^{pq}}''
    \f}

    This gives the following equation:
    
    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}
    \sqrt{m_i\,m_j}\left(\delta_{ij}\sum_{l=0}^{S-1}x_l\,\chi_{il}{B_{il1}^{pq}}'
    +x_j\,\chi_{ij}{B_{ij1}^{pq}}''\right)
    a_{q}^{(j)} &= \frac{15\,k_B\,T}{2} K_i\,\delta_{p1} & \text{for }i\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    The equations for \f$p=0\f$ do not determine the coefficients
    uniquely, so we introduce the following identity:

    \f{align*}{
    \sum_{i=0}^{S-1} \frac{\rho_i}{\rho}\,a_0^{(i)} = 0
    \f}

    Written in a suitable form:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1} \delta_{0q}\,x_j\,m_j\,a_q^{(j)} = 0
    \f}
    
    and we replace the \f$i=0,p=0\f$ equation with this.
  */
  void Solvea() {
    Eigen::MatrixXd A(_species.size() * _sonineOrder,_species.size() * _sonineOrder);
    Eigen::MatrixXd b (_species.size() * _sonineOrder, 1);
    
    for (size_t p = 0; p < _sonineOrder; p++)
      for (size_t i = 0; i < _species.size(); i++)
        for (size_t q = 0; q < _sonineOrder; q++)
          for (size_t j = 0; j < _species.size(); j++)
            {
              double sum = 0.0;
              if (i == j)
                for (size_t l = 0; l < _species.size(); l++)
                  sum += _species[l]._x * _gr(i, l) * Sd3o2(i,l,p,q);
              sum += _species[j]._x * _gr(i, j) * Sdd3o2(i,j,p,q);
              
              A(p * _species.size() + i, (q * _species.size())+j)
                = std::sqrt(_species[i]._mass * _species[j]._mass) * sum;
            }
    for (size_t p = 0; p < _sonineOrder; p++)
      for (size_t i = 0; i < _species.size(); i++)
        b(p * _species.size() + i) = 15 * _kT * K(i) * (p == 1) / 2.0;

    {//The additional identity
      const size_t p=0;
      const size_t i=0;
      for (size_t q = 0; q < _sonineOrder; q++)
	for (size_t j = 0; j < _species.size(); j++)
	  A(p * _species.size() + i, q * _species.size() + j)
	    = (q == 0) * _species[j]._x * _species[j]._mass;
      
      b(p * _species.size() + i) = 0;
    }
    
    _a = A.fullPivLu().solve(b);

    const double relerror = (A * _a - b).norm() / b.norm();
    if (relerror > 1e-6)
      std::cerr << "Warning! Solution for a did not converge!" << std::endl;

    if (_verbose || (relerror > 1e-6)) {
      std::cout << "Solving for a" << std::endl;
      std::cout << "A\n" << A << std::endl;
      std::cout << "b\n" << b << std::endl;
      std::cout << "x\n" << _a << std::endl;
      std::cout << "Relative error = " << relerror << std::endl;
    }
  }

  /*! \brief Solve for the Sonine coefficients \f$b_{q}^{(i)}\f$.
    
    Please see \ref Enskog::Solveh for more information on the decoding of the
    notation of Lopez de Haro, and how the solution is carried out.
    
    The linear equations to solve for the coefficients are (from Lopez
    de Haro et al (1983) Eq.(43)):

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}H_{ij}^{pq}\,b_{q}^{(j)} &= \frac{2}{k_B\,T} \frac{n_i^*}{n_i}\delta_{p0} & \text{for }i\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    where
    
    \f{multline*}{\scriptsize
    H_{ij}^{pq} = \frac{2}{5\,k_B\,T}\left(\delta_{ij}\sum_{l=0}^{S-1}\frac{n_i^*\,n_l^*}{n^2}\left[S_{5/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right),\,S_{5/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right)\right]'_{ij}\right.
    \\
    +\left.\frac{n_i^*\,n_j^*}{n^2} \left[S_{5/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right),\,S_{5/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right)\right]''_{ij}\right)
    \f}

    and \f$n_i^* = n_i\,K_i'\f$ and
    \f$\sigma_{ij}^{*2}=\chi_{ij}(K_i'\,K_j')^{-1}\sigma_{ij}^2\f$.

    Taking out the hidden dependence of the partial bracket integrals, we have:
    
    \f{align*}
    \left[S_{5/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right),\,S_{5/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right)\right]'_{ij} &= \chi_{ij}(K_i'\,K_j')^{-1}{B_{ij2}^{pq}}'
    \\
    \left[S_{5/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right),\,S_{5/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\frac{m}{2\,k_B\,T}\left(\bm{VV}-\frac{1}{3}V^2\bm{1}\right)\right]''_{ij} &= \chi_{ij}(K_i'\,K_j')^{-1}{B_{ij2}^{pq}}''
    \f}

    This gives the following equation:
    
    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}\left(\delta_{ij}\sum_{l=0}^{S-1}x_l \chi_{il}{B_{il2}^{pq}}'
    +x_j\,\chi_{ij}{B_{ij2}^{pq}}''\right)\,b_{q}^{(j)} &= 5\, K'_i\delta_{p0} & \text{for }i\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    Comparison against Xu and Stell
    -------------------------------

    Xu and Stell (1989), give the following equation for the Sonine coefficients:
    
    \f{align*}{
    \sum_{q=1}^{N-1}\left[b_q^{(i)}\sum_{j=0}^{S-1} \chi_{ij}\,x_j B^{pq'}_{ij2} + \sum_{j=0}^{S-1} \chi_{ij} x_j B^{pq''}_{ij2} b_{q}^{(j)}\right] = 5 H_i \delta_p0
    \f}

    where \f$H_i=K_{i}'\f$. Trying to frame their expression in terms
    of the Lopez de Haro et al form:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=1}^{N-1} \left[\delta_{ij}\sum_{l=0}^{S-1} x_l B^{pq'}_{il2}\chi_{il} + x_j B^{pq''}_{ij2}\chi_{ij}\right]b_q^{(j)} = 5\,K'_i \delta_p0
    \f}

    Clearly the two expressions are equivalent.
  */
  void Solveb() {
    Eigen::MatrixXd A(_species.size() * _sonineOrder, _species.size() * _sonineOrder);
    Eigen::MatrixXd b (_species.size() * _sonineOrder, 1);
    
    for (size_t p = 0; p < _sonineOrder; p++)
      for (size_t i = 0; i < _species.size(); i++)
        for (size_t q = 0; q < _sonineOrder; q++)
          for (size_t j = 0; j < _species.size(); j++)
            {
              double sum(0);
              if (i == j)
                for (size_t l = 0; l < _species.size(); l++)
                  sum += _species[l]._x * _gr(i, l) * Sd5o2(i,l,p,q);
              sum += _species[j]._x * _gr(i, j) * Sdd5o2(i,j,p,q);
	      
              A(p * _species.size() + i, q * _species.size() + j) = sum;
            }
    
    for (size_t p = 0; p < _sonineOrder; p++)
      for (size_t i = 0; i < _species.size(); i++)
        b(p * _species.size() + i) = 5 * Kd(i) * (p == 0);

    _b = A.fullPivLu().solve(b);

    const double relerror = (A * _b - b).norm() / b.norm();
    if (relerror > 1e-6)
      std::cerr << "Warning! Solution for b did not converge!" << std::endl;

    if (_verbose || (relerror > 1e-6)) {
      std::cout << "Solving for b" << std::endl;
      std::cout << "A\n" << A << std::endl;
      std::cout << "b\n" << b << std::endl;
      std::cout << "x\n" << _b << std::endl;
      std::cout << "Relative error = " << relerror << std::endl;
    }
  }
  
  /*! \brief Solve for the Sonine coefficients \f$d_{j,q}^{(k)}\f$.
    
    Please see \ref Enskog::Solveh for more information on the decoding of the
    notation of Lopez de Haro, and how the solution is carried out.
    
    The linear equations to solve for the coefficients are (from Lopez
    de Haro et al (1983) Eq.(43)):

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}\Lambda_{ij}^{pq}\,d_{j,q}^{(k)} &= \frac{8}{25\,k_B} \left(\delta_{ik} - \frac{\rho_i}{\rho}\right)\delta_{p0} & \text{for }i,k\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    where, as for the \f$a_{q}^{(j)}\f$, coefficients we have:

    \f{multline*}{\scriptsize
    \Lambda_{ij}^{pq} = \frac{8\sqrt{m_i\,m_j}}{75\,k_B^2\,T}\left(\delta_{ij}\sum_{l=0}^{S-1}\frac{n_i^*\,n_l^*}{n^2} \left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij}
    +\frac{n_i^*\,n_j^*}{n^2}\left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]''_{ij}\right)
    \f}

    but now \f$n_i^* = n_i\f$ and
    \f$\sigma_{ij}^{*2}=\chi_{ij}\sigma_{ij}^2\f$.

    Taking out the hidden dependence of the partial bracket integrals, we have:
    
    \f{align*}
    \left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij}  &= \chi_{ij}{B_{ij1}^{pq}}'
    \\
    \left[S_{3/2}^{(p)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V},\,S_{3/2}^{(q)}\left(\frac{m\,V^2}{2\,k_B\,T}\right)\left(\frac{m}{2\,k_B\,T}\right)^{1/2}\bm{V}\right]'_{ij} &= \chi_{ij}{B_{ij1}^{pq}}''
    \f}

    This gives the following equation:
    
    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1}  
    \sqrt{m_i\,m_j}\left(\delta_{ij}\sum_{l=0}^{S-1}x_i\,x_l\chi_{il}{B_{il1}^{pq}}'
    +x_i\,x_j\chi_{ij}{B_{ij1}^{pq}}''\right)
    d_{j,q}^{(k)} &= 3\, k_B\,T \left(\delta_{ik} - \frac{\rho_i}{\rho}\right)\delta_{p0} & \text{for }i,k\in[0,\,S-1],\,p\in[0,\,N-1]
    \f}

    The equations for \f$p=0\f$ do not determine the coefficients
    uniquely, so we introduce the following identity:

    \f{align*}{
    \sum_{i=0}^{S-1} \frac{\rho_i}{\rho}\,d_{i,0}^{(k)} = 0
    \f}

    Written in a suitable form:

    \f{align*}{
    \sum_{j=0}^{S-1}\sum_{q=0}^{N-1} \delta_{0q}\,x_j\,m_j\,\,d_{j,q}^{(k)}  = 0
    \f}
    
    and we replace the \f$i=0,p=0\f$ equation with this.
  */
  void Solved() {
    _d.resize(_species.size());
    
    for (size_t k(0); k < _species.size(); ++k) {
      Eigen::MatrixXd A(_species.size() * _sonineOrder, _species.size() * _sonineOrder);
      Eigen::MatrixXd b(_species.size() * _sonineOrder, 1);

      for (size_t p = 0; p < _sonineOrder; ++p)
        for (size_t i = 0; i < _species.size(); ++i)
	  for (size_t q = 0; q < _sonineOrder; ++q)
            for (size_t j = 0; j < _species.size(); ++j) {
              double sum(0);
              if (i == j)
                for (size_t l = 0; l < _species.size(); ++l)
                  sum += _species[i]._x * _species[l]._x * _gr(i, l) * Sd3o2(i,l,p,q);
              sum += _species[i]._x * _species[j]._x * _gr(i, j) * Sdd3o2(i,j,p,q);
              
              A(p * _species.size() + i, q * _species.size() + j) = sum * std::sqrt(_species[i]._mass * _species[j]._mass);
            }

      for (size_t p = 0; p < _sonineOrder; p++)
        for (size_t i = 0; i < _species.size(); i++)
	  b(p * _species.size() + i) = 3 * _kT * ((i == k) - _density * _species[i]._x * _species[i]._mass / rho()) * (p == 0);;

      {//The additional identity
	const size_t p = 0;
	const size_t i = 0;
	
	for (size_t q = 0; q < _sonineOrder; ++q)
	  for (size_t j = 0; j < _species.size(); j++)
	    A(p * _species.size() + i, q * _species.size() + j) = (q==0) * _species[j]._x * _species[j]._mass;	
	b(p * _species.size() + i) = 0;
      }
      
      _d[k] = A.fullPivLu().solve(b);
      
      // const double relerror = (A * _d[k] - b).norm() / b.norm();
      // if (relerror > 1e-6)
      // 	std::cerr << "Warning! Solution for d[" << k << "] did not converge!" << std::endl;
      
      // if (_verbose || (relerror > 1e-6)) {
      //   std::cout << "Solving for d[" << k << "]" << std::endl;
      //   std::cout << "A\n" << A << std::endl;
      //   std::cout << "b\n" << b << std::endl;
      //   std::cout << "x\n" << _d[k] << std::endl;
      // 	std::cout << "Relative error = " << relerror << std::endl;
      // }
    }
  }

  /*! \brief Generalised Sonine integral, \f${B_{ijl}^{pq}}'\f$.

    These are given by Lindenfeld and Shizgal (1979), and repeated by
    Xu and Stell.


    The original definition of Lindenfeld and Shizgal (1979) is:
    
    \f{align*}{
    M_{lnn'}^{(1)} &= -\frac{d^2\,Q}{2\,\pi}l!\sum_{p=0}^{\tilde{n}}\sum_{s=0}^{\tilde{n}-p}\sum_{m=0}^{\tilde{n}-p-s}\sum_{q=0}^{l}\sum_{r=0}^{l-q}4^p\frac{(r+s+p+q+1)!}{(p+q+1)!\,r!\,s!} \frac{\Gamma\left(n+n'-2\,s-2\,p-m+l-r-q-\frac{1}{2}\right)}{(n-m-s-p)!\,(n'-m-s-p)!\,(l-r-q)!\,m!}B^{(1)}_{pq}\,M_1^{l+p-r-q}\,M_2^{n+n'+q-2\,m-2\,s-p}\left(M_1-M_2\right)^{m+r+2\,s}
    \\
    M_{lnn'}^{(2)} &= -\frac{d^2\,Q}{2\,\pi}l!\,M_1^{n'+l/2}\,M_2^{n+l/2}\sum_{p=0}^{\tilde{n}}\sum_{q=0}^l 4^p\frac{\Gamma\left(n+n'-2\,p+l-q-\frac{1}{2}\right)}{(n-p)!\,(n'-p)!\,(l-q)!} B_{pq}^{(2)}(\infty)
    \f}

    where:

    \f{align*}{
    Q&=\sqrt{\frac{2\,k_B\,T}{\mu_{ij}}} & &
    \\
    B_{pq}^{(1)}(\infty) &= \frac{(2\,p+q+1)!}{2\,q!\,(2\,p+1)!}-2^{q-1}\frac{(p+q+1)!}{p!\,q!}
    &
    B_{pq}^{(2)}(\infty) &= \frac{(2\,p+q+1)!}{2\,q!(2\,p+1)!} - \frac{1}{2}\delta_{p0}\,\delta_{q0}
    \f}

    Trying to write this in terms of Xu and Stell:
    
    \f{align*}{
    M_{lnn'}^{(1)} &= -\frac{\sigma_{ij}^2}{2\,\pi}\sqrt{\frac{2\,k_B\,T}{m_i}}l!\sum_{p=0}^{\tilde{n}}\sum_{s=0}^{\tilde{n}-p}\sum_{m=0}^{\tilde{n}-p-s}\sum_{q=0}^{l}\sum_{r=0}^{l-q}4^p\frac{(r+s+p+q+1)!}{(p+q+1)!\,r!\,s!} \frac{\Gamma\left(n+n'-2\,s-2\,p-m+l-r-q-\frac{1}{2}\right)}{(n-m-s-p)!\,(n'-m-s-p)!\,(l-r-q)!\,m!}B^{(1)}_{pq}\,M_{ij}^{l+p-r-q}\,M_{ji}^{n+n'+q-2\,m-2\,s-p-1/2}\left(M_{ij}-M_{ji}\right)^{m+r+2\,s}
    \\
    M_{lnn'}^{(2)} &= -\frac{\sigma_{ij}^2}{2\,\pi}\sqrt{\frac{2\,k_B\,T}{m_i}}l!\,M_{ij}^{n'+l/2}\,M_{ji}^{n+(l-1)/2}\sum_{p=0}^{\tilde{n}}\sum_{q=0}^l 4^p\frac{\Gamma\left(n+n'-2\,p+l-q-\frac{1}{2}\right)}{(n-p)!\,(n'-p)!\,(l-q)!} B_{pq}^{(2)}(\infty)
    \f}
    
    To get the same notation as Xu and Stell, we make the relabeling as follows:
    \f$\tilde{n}\to \tilde{p},\,n\to p,\,n'\to q,\,p\to n,\,q\to t\f$
    However, we still need to apply the effects of Eq.(5) from Lindenfeld and Shizgal.
   */
  double Bpqdijl(int i, int j, int p, int q, int l) const {
    const int ptilde = std::min(p, q);

    if (l>2)
      throw std::runtime_error("l>2");
    
    double sum(0);

    //Must use integers to prevent underflow!
    for (int n(0); n <= ptilde; ++n)
      for (int s(0); s <= (ptilde - n); ++s)
	for (int m(0); m <= (ptilde - n - s); ++m)
 	  for (int t(0); t <= l; ++t)
	    for (int r(0); r <= (l-t); ++r)
	      sum += std::pow(4, n)
		* (std::tgamma(r + s + n + t + 2) / std::tgamma(n + t + 2) / std::tgamma(r + 1) / std::tgamma(s + 1))
		* (std::tgamma(p + q - 2 * s - 2 * n - m + l - r - t - 0.5) / std::tgamma(p - m - s - n + 1) / std::tgamma(q - m - s - n + 1) / std::tgamma(l - r - t + 1) / std::tgamma(m + 1))
		//B^{(1)}_{nt}
		* (std::tgamma(2 * n + t + 2) / 2 / std::tgamma(t + 1) / std::tgamma(2 * n + 2) - std::pow(2, t - 1) * std::tgamma(n+t+2) / std::tgamma(n+1) / std::tgamma(t+1))
                //Cont.
		* std::pow(M(i,j), l+n-r-t) * std::pow(M(j, i), p+q+t-2*m-2*s-n-0.5) * std::pow(M(i,j) - M(j,i), m+r+2*s);
    
    const double Al = (l == 2) ? (2.0 / 3.0) : 1.0; 

    return 2 * std::pow(sigma(i,j), 2) * std::sqrt(2 * _kT / _species[i]._mass) * Al * std::tgamma(l + 1) *  sum;
  }

  /*! \brief Generalised Sonine integral, \f${B_{ijl}^{pq}}''\f$.
   */
  double Bpqddijl(int i, int j, int p, int q, int l) const {
    const int ptilde = std::min(p, q);

    if (l>2)
      throw std::runtime_error("l>2");
    
    const double Al = (l == 2) ? (2.0 / 3) : 1.0; 

    double sum(0);

    for (int n(0); n <= ptilde; ++n)
      for (int t(0); t <= l; ++t)
        sum += std::pow(4, n)
          * std::tgamma(p + q - 2 * n + l - t - 0.5) / std::tgamma(p - n + 1) / std::tgamma(q - n + 1) / std::tgamma(l - t + 1)
          //B_{nt}^{(2)}
          * (std::tgamma(2 * n + t + 2) / 2 / std::tgamma(t + 1) / std::tgamma(2 * n + 2) - 0.5 * (n==0) * (t==0))
          ;
          
    return 2 * std::pow(sigma(i,j), 2) * std::sqrt(2 * _kT / _species[i]._mass) * Al * std::tgamma(l+1) * std::pow(M(i,j), q + l * 0.5) * std::pow(M(j,i), p + (l - 1) * 0.5) * sum;
  }

  double Bpqdij0(int i, int j, int p, int q) const {
    const int ptilde = std::min(p,q);
    
    double sum(0);
    for (int n(0); n <= ptilde; ++n)
      for (int s(0); s <= (ptilde - n); ++s)
	for (int m(0); m <= (ptilde - n - s); ++m)
	  sum += std::pow(4, n)
	    * (std::tgamma(s+n+2) / (std::tgamma(n+2) * std::tgamma(s+1)))
	    * (std::tgamma(p + q - 2 * s - 2 * n - m - 0.5) / (std::tgamma(p - m - s - n + 1) * std::tgamma(q - m - s - n + 1) * std::tgamma(m + 1)))
	    * (-n * 0.5)
	    * std::pow(M(i,j), n) * std::pow(M(j,i), p + q - 2 * m - 2 * s - n - 0.5) * std::pow(M(i,j) - M(j,i), m + 2 * s)
	    ;

    return 2 * std::pow(sigma(i, j), 2) * std::sqrt(2 * _kT / _species[i]._mass) * sum;
  }

  double Bpqddij0(int i, int j, int p, int q) const {
    double sum(0);
    const int ptilde = std::min(p,q);

    for (int n(0); n <= ptilde; ++n)
      sum += std::pow(4, n) * std::tgamma(p + q - 2 * n - 0.5) / (std::tgamma(p - n + 1) * std::tgamma(q - n + 1)) * 0.5 * (1 - (n==0));
    
    return 2 * std::pow(sigma(i, j), 2) * std::sqrt(2 * _kT / _species[i]._mass)
      * std::pow(M(i,j), q) * std::pow(M(j,i), p - 0.5)
      * sum;
  }
};
