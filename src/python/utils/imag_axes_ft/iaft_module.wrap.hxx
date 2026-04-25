#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_iaft_module_GUARDS
#define C2PY_HXX_DECLARATION_iaft_module_GUARDS
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::ir::IR> = true;
template <>
inline constexpr auto c2py::tp_name<imag_axes_ft::ir::IR> = "iaft_module.IR";
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::IAFT> = true;
template <>
inline constexpr auto c2py::tp_name<imag_axes_ft::IAFT> = "iaft_module.IAFT";
#endif