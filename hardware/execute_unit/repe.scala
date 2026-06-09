package execute_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

import exp_unit.ExpUnitFixPoint

class RePE(
    bits: Int,
    point: Int,
    regWidth: Int, // number of column regs feeding this PE
    colSelectBits: Int,
    id: (Int, Int)
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val q_in = Input(fpType)               // row input (from Q buffer / reg chain)
    val reg_cols = Input(Vec(regWidth, fpType)) // column register inputs
    val out = Output(fpType)

    val sel_col = Input(UInt(colSelectBits.W))
    val acc_ctrl = Input(UInt(2.W))
    val exp_ctrl = Input(UInt(2.W))
  })

  val acc_clear :: acc_idle :: acc_accumulate :: acc_move_out :: Nil = Enum(4)
  val exp_idle :: exp_compute :: Nil = Enum(2)

  val exp_unit = Module(new ExpUnitFixPoint(bits, point, 6, 4))

  val a = Wire(fpType)      // row source
  val b = Wire(fpType)      // selected column
  val mul = Wire(fpType)    // product
  val acc = Reg(fpType)
  val score_exp = Reg(fpType)

  io.out := score_exp

  a := io.q_in
  switch(io.acc_ctrl) {
    is(acc_move_out) { a := score_exp }
  }

  val col_vec = Wire(Vec(regWidth + 1, fpType))
  for (i <- 0 until regWidth) col_vec(i) := io.reg_cols(i)
  col_vec(regWidth) := FixedPoint(0, bits.W, point.BP)

  var selW = when(io.sel_col === 0.U) {
    b := col_vec(0)
  }
  for (i <- 1 until (regWidth + 1)) {
    selW = selW.elsewhen(io.sel_col === i.U) {
      b := col_vec(i)
    }
  }
  selW.otherwise {
    b := col_vec(regWidth)
  }

  mul := a * b

  switch(io.acc_ctrl) {
    is(acc_clear) { acc := FixedPoint(0, bits.W, point.BP) }
    is(acc_idle)  { acc := acc }
    is(acc_accumulate) { acc := acc + mul }
    is(acc_move_out) {
      acc := mul
      io.out := acc
    }
  }

  exp_unit.io.in_value := FixedPoint(0, bits.W, point.BP)
  switch(io.exp_ctrl) {
    is(exp_idle) { score_exp := score_exp }
    is(exp_compute) {
      exp_unit.io.in_value := acc
      score_exp := exp_unit.io.out_exp
    }
  }
}