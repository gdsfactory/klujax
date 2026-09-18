//! Internal proc-macros for `klujax-ffi`.
//!
//! `#[xla_handler]` turns an ordinary safe function over [`klujax_ffi::call_frame::Buf`]
//! / [`klujax_ffi::call_frame::BufMut`] into an XLA typed-FFI handler: it emits
//! the `#[no_mangle] pub unsafe extern "C"` wrapper, decodes each `Buf`
//! argument (in order) and each `BufMut` result (in order) from the
//! `XLA_FFI_CallFrame`, and dispatches to the function.
//!
//! ```ignore
//! #[xla_handler(scalars(f64 => "solve_f64", C64 => "solve_c128"))]
//! fn solve<T: Scalar>(ai: Buf<i32>, aj: Buf<i32>, ax: Buf<T>, b: Buf<T>, x: BufMut<T>)
//!     -> Result<(), ErrorInfo> { ... }
//! ```
//!
//! Paths in the generated code are `crate::...`, so this macro is only usable
//! from within the `klujax-ffi` crate.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::visit_mut::VisitMut;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, FnArg, Ident, ItemFn, Pat, Token, Type, TypePath,
};

mod kw {
    syn::custom_keyword!(export);
    syn::custom_keyword!(scalars);
}

/// Attribute arguments: either `export = "target"` or
/// `scalars(Concrete => "target", ...)`.
struct Args {
    export: Option<String>,
    scalars: Vec<(Type, String)>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut export = None;
        let mut scalars = Vec::new();

        if input.peek(kw::export) {
            input.parse::<kw::export>()?;
            input.parse::<Token![=]>()?;
            export = Some(input.parse::<syn::LitStr>()?.value());
        } else if input.peek(kw::scalars) {
            input.parse::<kw::scalars>()?;
            let body;
            syn::parenthesized!(body in input);
            while !body.is_empty() {
                let ty: Type = body.parse()?;
                body.parse::<Token![=>]>()?;
                let name = body.parse::<syn::LitStr>()?.value();
                scalars.push((ty, name));
                if body.peek(Token![,]) {
                    body.parse::<Token![,]>()?;
                }
            }
        } else if !input.is_empty() {
            return Err(input.error("expected `export = \"...\"` or `scalars(...)`"));
        }

        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
        if export.is_none() && scalars.is_empty() {
            return Err(input.error("`#[xla_handler]` needs `export = \"...\"` or `scalars(...)`"));
        }
        Ok(Self { export, scalars })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Arg,
    Ret,
}

struct Param {
    name: Ident,
    ty: Type,
    kind: Kind,
    index: usize,
}

fn type_tail(ty: &Type) -> Option<&Ident> {
    match ty {
        Type::Path(TypePath { qself: None, path }) => path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

/// Substitute `from` (a generic type parameter) with `to`, recursively.
#[derive(Clone)]
struct Subst {
    from: Ident,
    to: Type,
}

impl VisitMut for Subst {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        if let Type::Path(p) = ty {
            if p.qself.is_none() && p.path.is_ident(&self.from) {
                *ty = self.to.clone();
                return;
            }
        }
        syn::visit_mut::visit_type_mut(self, ty);
    }
}

fn collect_params(func: &ItemFn) -> syn::Result<Vec<Param>> {
    let mut params = Vec::new();
    let (mut arg_i, mut ret_i) = (0usize, 0usize);
    for input in &func.sig.inputs {
        let FnArg::Typed(pat_type) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "`#[xla_handler]` functions cannot take `self`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "`#[xla_handler]` parameters must be simple identifiers",
            ));
        };
        let ty = (*pat_type.ty).clone();
        let kind = match type_tail(&ty).map(Ident::to_string).as_deref() {
            Some("Buf") => Kind::Arg,
            Some("BufMut") => Kind::Ret,
            _ => {
                return Err(syn::Error::new_spanned(
                    &ty,
                    "`#[xla_handler]` parameters must be `Buf<...>` (argument) or `BufMut<...>` (result)",
                ))
            }
        };
        let index = match kind {
            Kind::Arg => {
                let i = arg_i;
                arg_i += 1;
                i
            }
            Kind::Ret => {
                let i = ret_i;
                ret_i += 1;
                i
            }
        };
        params.push(Param {
            name: pat_ident.ident.clone(),
            ty,
            kind,
            index,
        });
    }
    Ok(params)
}

fn expand(args: Args, func: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &func,
            "`#[xla_handler]` cannot be async",
        ));
    }
    let params = collect_params(&func)?;
    let fn_name = &func.sig.ident;
    let arg_names: Vec<&Ident> = params.iter().map(|p| &p.name).collect();

    // Which concrete monomorphizations to emit. `export` => one, no generics.
    let generic_ty: Option<Ident> = func
        .sig
        .generics
        .type_params()
        .next()
        .map(|p| p.ident.clone());
    let mut expansions: Vec<(Option<Type>, String)> = Vec::new();
    if let Some(export) = &args.export {
        expansions.push((None, export.clone()));
    }
    for (ty, name) in &args.scalars {
        expansions.push((Some(ty.clone()), name.clone()));
    }

    let mut wrappers = Vec::new();
    for (concrete, export) in expansions {
        // Types with the generic parameter replaced by `concrete`.
        let subst = match (&generic_ty, &concrete) {
            (Some(from), Some(to)) => Some(Subst {
                from: from.clone(),
                to: to.clone(),
            }),
            _ => None,
        };
        let typed: Vec<Type> = params
            .iter()
            .map(|p| {
                let mut ty = p.ty.clone();
                if let Some(mut subst) = subst.clone() {
                    subst.visit_type_mut(&mut ty);
                }
                ty
            })
            .collect();

        let decode = params.iter().zip(&typed).map(|(p, ty)| {
            let name = &p.name;
            let idx = p.index;
            let what = name.to_string();
            let (trait_path, method) = match p.kind {
                Kind::Arg => (quote!(crate::call_frame::DecodeArg), quote!(decode_arg)),
                Kind::Ret => (quote!(crate::call_frame::DecodeRet), quote!(decode_ret)),
            };
            quote! {
                let #name: #ty = <#ty as #trait_path>::#method(frame, #idx, #what)?;
            }
        });

        let call = match &concrete {
            Some(ty) => quote! { #fn_name::<#ty>( #(#arg_names),* ) },
            None => quote! { #fn_name( #(#arg_names),* ) },
        };

        let export_ident = Ident::new(&export, Span::call_site());
        let safety = "`frame_ptr` is the valid, non-null call frame XLA passes to this handler, alive for the duration of the call; the `Buf`/`BufMut` decode checks each argument/result.";
        wrappers.push(quote! {
            #[no_mangle]
            pub unsafe extern "C" fn #export_ident(
                frame_ptr: *mut crate::xla_ffi::XLA_FFI_CallFrame,
            ) -> *mut crate::xla_ffi::XLA_FFI_Error {
                #[doc = ""]
                #[doc = "# Safety"]
                #[doc = ""]
                #[doc = #safety]
                unsafe {
                    // SAFETY: see the `# Safety` doc on this function.
                    crate::error::guard(frame_ptr, || {
                        let frame = &crate::call_frame::Frame::from_raw(frame_ptr);
                        #(#decode)*
                        #call
                    })
                }
            }
        });
    }

    Ok(quote! {
        #func
        #(#wrappers)*
    })
}

/// Turn a safe `Buf`/`BufMut` function into an XLA FFI handler.
#[proc_macro_attribute]
pub fn xla_handler(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as Args);
    let func = parse_macro_input!(item as ItemFn);
    expand(args, func)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
