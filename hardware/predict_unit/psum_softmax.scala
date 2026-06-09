package predict_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class FixedPointDiv(bitWidth: Int, point: Int) extends Module {
  require(bitWidth > 0 && point >= 0)
  val fpType = FixedPoint(bitWidth.W, point.BP)
  val io = IO(new Bundle {
    val numerator = Input(fpType)
    val denominator = Input(fpType)
    val quotient = Output(fpType)
  })

  val numS = io.numerator.asSInt
  val denS = io.denominator.asSInt

  val shiftedNum = (numS << point)
  val resultWidth = bitWidth + point + 1
  val resultS = Wire(SInt(resultWidth.W))

  when(denS === 0.S) {
    resultS := 0.S
  } .otherwise {
    resultS := (shiftedNum / denS).resize(resultWidth)
  }

  io.quotient := resultS.asFixedPoint(point.BP).asTypeOf(fpType)
}

class PSumSoftmax(bitWidth: Int, point: Int) extends Module {
  require(bitWidth > 0 && point >= 0)
  val fpType = FixedPoint(bitWidth.W, point.BP)
  val io = IO(new Bundle {
    val add1 = Input(fpType)
    val add2 = Input(fpType)
    val exp = Input(fpType)
    val out = Output(fpType)
    val en = Output(Bool())
    val s = Input(UInt(1.W))
  })

  val sum = Wire(fpType)
  sum := io.add1 + io.add2

  val divider = Module(new FixedPointDiv(bitWidth, point))
  divider.io.numerator := sum
  divider.io.denominator := io.exp

  when(io.s === 0.U) {
    io.out := sum
    io.en := false.B
  } .otherwise {
    when(io.exp === 0.F(bitWidth.W, point.BP)) {
      io.out := 0.F(bitWidth.W, point.BP)
      io.en := false.B
    } .otherwise {
      io.out := divider.io.quotient
      io.en := true.B
    }
  }
}