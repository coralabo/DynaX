package execute_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class RePEArray(
    peCountPerRow: Int,
    bits: Int,
    point: Int,
    regWidth: Int,
    numRows: Int,
    colSelectBits: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val q_left_vec = Input(Vec(numRows, fpType))
    val regs_top = Input(Vec(regWidth, fpType))
    val rows_adder_out = Output(Vec(numRows, fpType))

    val clr = Input(Bool())
    val sel_cols = Input(Vec(numRows, Vec(peCountPerRow, UInt(colSelectBits.W))))
    val acc_ctrl = Input(UInt(2.W))
    val exp_ctrl = Input(UInt(2.W))
  })

  val rows = for (i <- 0 until numRows) yield {
    Module(new RePERow(peCountPerRow, bits, point, regWidth, colSelectBits, i))
  }

  val (acc_clear, acc_idle, acc_accumulate, acc_move_out) = (
    rows(0).acc_clear,
    rows(0).acc_idle,
    rows(0).acc_accumulate,
    rows(0).acc_move_out
  )
  val (exp_idle, exp_compute) = (rows(0).exp_idle, rows(0).exp_compute)

  for (i <- 0 until numRows) {
    rows(i).io.clr := io.clr
    rows(i).io.sel_cols := io.sel_cols(i)
    rows(i).io.acc_ctrl := io.acc_ctrl
    rows(i).io.exp_ctrl := io.exp_ctrl
    rows(i).io.q_left := io.q_left_vec(i)
    io.rows_adder_out(i) := rows(i).io.adder_out
  }

  rows(0).io.regs_top := io.regs_top
  for (i <- 0 until numRows - 1)
    rows(i + 1).io.regs_top := rows(i).io.regs_bottom
}