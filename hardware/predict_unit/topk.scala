package predict_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

class TopFirst(bits: Int, point: Int, idxBits: Int, depth: Int) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val inData = Input(fpType)
    val outData = Output(fpType)
    val maxIdx = Output(UInt(idxBits.W))
    val outIdx = Output(UInt(idxBits.W))
    val outValid = Output(Bool())
  })

  val maxReg = RegInit(FixedPoint.fromDouble(0.0, bits.W, point.BP))
  val runnerUpReg = RegInit(FixedPoint.fromDouble(0.0, bits.W, point.BP))
  val posCounter = RegInit(0.U(idxBits.W))
  val maxIndex = RegInit(0.U(idxBits.W))
  val runnerUpIndex = RegInit(0.U(idxBits.W))
  val validPulse = RegInit(false.B)

  val curPos = posCounter

  when(io.enable) {
    when(io.inData > maxReg) {
      runnerUpReg := maxReg
      runnerUpIndex := maxIndex
      maxReg := io.inData
      maxIndex := curPos
    }.otherwise {
      runnerUpReg := io.inData
      runnerUpIndex := curPos
    }

    posCounter := posCounter + 1.U

    when(posCounter === (depth - 1).U) {
      posCounter := 0.U
      maxReg := FixedPoint.fromDouble(0.0, bits.W, point.BP)
      runnerUpReg := FixedPoint.fromDouble(0.0, bits.W, point.BP)
      validPulse := true.B
    }.otherwise {
      validPulse := false.B
    }
  }.otherwise {
    validPulse := false.B
  }

  io.outData := runnerUpReg
  io.outIdx := runnerUpIndex
  io.maxIdx := maxIndex
  io.outValid := validPulse
}

class TopStage(bits: Int, point: Int, idxBits: Int) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val inIdxValid = Input(Bool())
    val inIdx = Input(UInt(idxBits.W))
    val inData = Input(fpType)
    val outData = Output(fpType)
    val maxIdx = Output(UInt(idxBits.W))
    val outIdx = Output(UInt(idxBits.W))
    val outValid = Output(Bool())
  })

  val maxReg = RegInit(FixedPoint.fromDouble(0.0, bits.W, point.BP))
  val runnerUpReg = RegInit(FixedPoint.fromDouble(0.0, bits.W, point.BP))
  val maxIndex = RegInit(0.U(idxBits.W))
  val runnerUpIndex = RegInit(0.U(idxBits.W))
  val validReg = RegInit(false.B)

  when(io.inData > maxReg) {
    runnerUpReg := maxReg
    runnerUpIndex := maxIndex
    maxReg := io.inData
    maxIndex := io.inIdx
  }.otherwise {
    runnerUpReg := io.inData
    runnerUpIndex := io.inIdx
  }

  when(io.inIdxValid) {
    maxReg := FixedPoint.fromDouble(0.0, bits.W, point.BP)
    runnerUpReg := FixedPoint.fromDouble(0.0, bits.W, point.BP)
  }

  validReg := io.inIdxValid

  io.outData := runnerUpReg
  io.outIdx := runnerUpIndex
  io.maxIdx := maxIndex
  io.outValid := validReg
}

class TopK(m: Int, n: Int, bits: Int, point: Int) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val idxBits = log2Ceil(m max 2)
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val inData = Input(fpType)
    val outData = Output(Vec(n, fpType))
    val idx = Output(Vec(n, UInt(idxBits.W)))
    val idxValid = Output(Vec(n, Bool()))
  })

  val first = Module(new TopFirst(bits, point, idxBits, m))
  val stages = Seq.fill(n - 1)(Module(new TopStage(bits, point, idxBits)))

  first.io.inData := io.inData
  first.io.enable := io.enable

  if (stages.nonEmpty) {
    stages(0).io.inData := first.io.outData
    stages(0).io.inIdx := first.io.outIdx
    stages(0).io.inIdxValid := first.io.outValid

    for (i <- 1 until stages.length) {
      stages(i).io.inData := stages(i - 1).io.outData
      stages(i).io.inIdx := stages(i - 1).io.outIdx
      stages(i).io.inIdxValid := stages(i - 1).io.outValid
    }
  }

  io.outData(0) := first.io.outData
  io.idx(0) := first.io.maxIdx
  io.idxValid(0) := first.io.outValid

  for (i <- 1 until n) {
    io.outData(i) := stages(i - 1).io.outData
    io.idx(i) := stages(i - 1).io.maxIdx
    io.idxValid(i) := stages(i - 1).io.outValid
  }
}