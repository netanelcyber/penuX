option casemap:none
extern printf : proc
extern ExitProcess : proc
include \masm32\include\kernel32.inc

.data
fmt_val db "%s: %.18Lf",10,0
lbl_orig db "LabValue",0
lbl_hi db "ID High (scaled)",0
lbl_lo db "ID Low (scaled)",0
lbl_cf db "CF5",0
lbl_gate db "Gate",0

csv_file db "mimic_subset.csv",0
buffer db 1024 dup(0)       ; buffer לקריאת שורה
row_data db 3*24 dup(0)     ; labValue + id_hi + id_lo (float384)

; float384 / x87
labValue dt 0.0
cf_a0 dt 0.0
cf_a1 dt 0.0
cf_a2 dt 0.0
cf_a3 dt 0.0
cf_a4 dt 0.0
cf_result dt 0.0
gate_result dt 0.0
id_hi dt 0.0
id_lo dt 0.0
id_hi_scaled dt 0.0
id_lo_scaled dt 0.0

tmp_min dt 0.0
tmp_max dt 0.0

one dt 1.0
zero dt 0.0
ln2 dt 0.6931471805599453094

.code

; -----------------------
; sigmoid on ST0
; -----------------------
sigmoid proc
    fchs
    fldln2
    fxch
    fyl2x
    fld1
    fscale
    fstp st(1)
    fld1
    fadd
    fld1
    fdivrp
    ret
sigmoid endp

; -----------------------
; CF5 evaluation
; -----------------------
cf5_eval proc
    fld st(0)
    frndint
    fstp tbyte ptr [cf_a0]
    fsub
    fld1
    fdivrp

    fld st(0)
    frndint
    fstp tbyte ptr [cf_a1]
    fsub
    fld1
    fdivrp

    fld st(0)
    frndint
    fstp tbyte ptr [cf_a2]
    fsub
    fld1
    fdivrp

    fld st(0)
    frndint
    fstp tbyte ptr [cf_a3]
    fsub
    fld1
    fdivrp

    frndint
    fstp tbyte ptr [cf_a4]

    fld tbyte ptr [cf_a4]
    fld1
    fdivrp
    fadd tbyte ptr [cf_a3]
    fld1
    fdivrp
    fadd tbyte ptr [cf_a2]
    fld1
    fdivrp
    fadd tbyte ptr [cf_a1]
    fld1
    fdivrp
    fadd tbyte ptr [cf_a0]

    fstp tbyte ptr [cf_result]
    ret
cf5_eval endp

; -----------------------
; MinMax scaling ST0 -> store
; ST0 = value
; tmp_min, tmp_max set externally
; store result to dest
; -----------------------
minmax_scale proc dest:QWORD
    fsub tbyte ptr [tmp_min]
    fld tbyte ptr [tmp_max]
    fsub tbyte ptr [tmp_min]
    fdivrp
    fstp tbyte ptr [dest]
    ret
minmax_scale endp

; -----------------------
; convert ASCII digits to float (simple, single number)
; in: rcx = buffer ptr, out: ST0 = value
; -----------------------
ascii_to_float proc
    xor rax, rax
    xor rbx, rbx
    xor rcx, rcx
    ; for demo: hardcode 2.718
    fld tbyte ptr [one]        ; ST0 = 1.0
    fadd tbyte ptr [ln2]       ; 1 + 0.693 ~ 1.693
    fadd tbyte ptr [one]       ; just demo
    ret
ascii_to_float endp

; -----------------------
; main
; -----------------------
main proc
    sub rsp, 128

    ; Demo: Load CSV batch (1 line)
    ; Simplified: assume we read 1 row: labValue,id_hi,id_lo
    ; Normally ReadFile + parsing

    ; labValue
    call ascii_to_float
    fstp tbyte ptr [labValue]

    ; id_hi
    fld1
    fstp tbyte ptr [id_hi]

    ; id_lo
    fld1
    fstp tbyte ptr [id_lo]

    ; Setup MinMax (demo)
    fldz
    fstp tbyte ptr [tmp_min]
    fld1
    fstp tbyte ptr [tmp_max]

    ; scale hi/lo
    fld tbyte ptr [id_hi]
    lea rax, id_hi_scaled
    call minmax_scale

    fld tbyte ptr [id_lo]
    lea rax, id_lo_scaled
    call minmax_scale

    ; CF5
    fld tbyte ptr [labValue]
    call cf5_eval

    ; Gate
    fld tbyte ptr [cf_result]
    fadd tbyte ptr [id_hi_scaled]
    fadd tbyte ptr [id_lo_scaled]
    call sigmoid
    fstp tbyte ptr [gate_result]

    ; Output
    lea rcx, fmt_val
    lea rdx, lbl_orig
    fld tbyte ptr [labValue]
    sub rsp, 16
    fstp tbyte ptr [rsp]
    lea r8, [rsp]
    call printf
    add rsp, 16

    lea rcx, fmt_val
    lea rdx, lbl_hi
    fld tbyte ptr [id_hi_scaled]
    sub rsp, 16
    fstp tbyte ptr [rsp]
    lea r8, [rsp]
    call printf
    add rsp, 16

    lea rcx, fmt_val
    lea rdx, lbl_lo
    fld tbyte ptr [id_lo_scaled]
    sub rsp, 16
    fstp tbyte ptr [rsp]
    lea r8, [rsp]
    call printf
    add rsp, 16

    lea rcx, fmt_val
    lea rdx, lbl_cf
    fld tbyte ptr [cf_result]
    sub rsp, 16
    fstp tbyte ptr [rsp]
    lea r8, [rsp]
    call printf
    add rsp, 16

    lea rcx, fmt_val
    lea rdx, lbl_gate
    fld tbyte ptr [gate_result]
    sub rsp, 16
    fstp tbyte ptr [rsp]
    lea r8, [rsp]
    call printf
    add rsp, 16

    add rsp, 128
    xor eax, eax
    call ExitProcess
main endp

end