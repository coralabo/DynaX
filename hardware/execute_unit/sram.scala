package execute_unit

import chisel3._
import chisel3.util._

class SRAMBank(val depth: Int, val width: Int) extends Module {
  val io = IO(new Bundle {
    val addr = Input(UInt(log2Ceil(depth).W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
    val writeEnable = Input(Bool())
  })

  val mem = SyncReadMem(depth, UInt(width.W))

  when(io.writeEnable) {
    mem.write(io.addr, io.dataIn)
  }

  // synchronous read (read enable = !writeEnable to avoid read during write)
  io.dataOut := mem.read(io.addr, !io.writeEnable)
}

class SRAM(val bankCount: Int, val bankDepth: Int, val bankWidth: Int) extends Module {
  val bankIdxBits = log2Ceil(bankCount)
  val bankOffBits = log2Ceil(bankDepth)
  val addrWidth = bankIdxBits + bankOffBits

  val io = IO(new Bundle {
    val addr = Input(UInt(addrWidth.W))
    val dataIn = Input(UInt(bankWidth.W))
    val dataOut = Output(UInt(bankWidth.W))
    val writeEnable = Input(Bool())
  })

  // create banks
  val banks = Seq.tabulate(bankCount)(i => Module(new SRAMBank(bankDepth, bankWidth)))

  // slice address: high bits = bank index, low bits = offset within bank
  val bankAddr = io.addr(addrWidth - 1, bankOffBits)   // bank index
  val bankOffset = io.addr(bankOffBits - 1, 0)         // address inside bank

  // connect each bank (only selected bank receives writeEnable)
  for (i <- 0 until bankCount) {
    banks(i).io.addr := bankOffset
    banks(i).io.dataIn := io.dataIn
    banks(i).io.writeEnable := io.writeEnable && (bankAddr === i.U)
  }

  // select dataOut from active bank
  val selVec = Seq.tabulate(bankCount)(i => (bankAddr === i.U) -> banks(i).io.dataOut)
  io.dataOut := Mux1H(selVec, 0.U(bankWidth.W))
}