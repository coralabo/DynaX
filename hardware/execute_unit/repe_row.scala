package execute_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class RePERow(
    peCount: Int,
    bits: Int,
    point: Int,
    regWidth: Int,
    colSelectBits: Int,
    rowId: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val q_left = Input(fpType)
    val regs_top = Input(Vec(regWidth, fpType))
    val regs_bottom = Output(Vec(regWidth, fpType))
    val adder_out = Output(fpType)

    val clr = Input(Bool())
    val sel_cols = Input(Vec(peCount, UInt(colSelectBits.W)))
    val acc_ctrl = Input(UInt(2.W))
    val exp_ctrl = Input(UInt(2.W))
  })

  val pes = for (i <- 0 until peCount) yield {
    Module(new RePE(bits, point, regWidth, colSelectBits, (rowId, i)))
  }

  val (acc_clear, acc_idle, acc_accumulate, acc_move_out) = (
    pes(0).acc_clear,
    pes(0).acc_idle,
    pes(0).acc_accumulate,
    pes(0).acc_move_out
  )
  val (exp_idle, exp_compute) = (pes(0).exp_idle, pes(0).exp_compute)

  val reg_q = Reg(fpType)
  val reg_cols = Reg(Vec(regWidth, fpType))

  val row_sum = Reg(fpType)
  io.adder_out := row_sum

  for (i <- 0 until peCount) {
    pes(i).io.q_in := reg_q
    pes(i).io.reg_cols := reg_cols
    pes(i).io.sel_col := io.sel_cols(i)
    pes(i).io.acc_ctrl := io.acc_ctrl
    pes(i).io.exp_ctrl := io.exp_ctrl
  }

  for (i <- 0 until regWidth) io.regs_bottom(i) := reg_cols(i)

  when(io.clr) {
    reg_q := FixedPoint(0, bits.W, point.BP)
    for (i <- 0 until regWidth) reg_cols(i) := FixedPoint(0, bits.W, point.BP)
    row_sum := FixedPoint(0, bits.W, point.BP)
  }.otherwise {
    reg_q := io.q_left
    reg_cols := io.regs_top
    io.regs_bottom := reg_cols

    row_sum := pes.map(_.io.out).reduce(_ + _)
  }
}