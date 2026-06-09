package predict_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint
import exp_unit.ExpUnitFixPoint

class PrePE_1_2(outBits: Int) extends Module {
  val io = IO(new Bundle {
    val left_in = Input(UInt(4.W))
    val sel_in = Input(UInt(1.W))
    val top_in0 = Input(UInt(4.W))
    val top_in1 = Input(UInt(4.W))

    val right_out = Output(UInt(4.W))
    val sel_out = Output(UInt(1.W))
    val bottom_out0 = Output(UInt(4.W))
    val bottom_out1 = Output(UInt(4.W))

    val psum_in = Input(UInt(outBits.W))
    val psum_out = Output(UInt(outBits.W))

    val state = Input(UInt(2.W))
  })

  val sIdle :: sClear :: sCalc :: sInput :: Nil = Enum(4)

  val topReg0 = Reg(UInt(4.W))
  val topReg1 = Reg(UInt(4.W))
  val leftReg = Reg(UInt(4.W))
  val selReg = Reg(UInt(1.W))
  val psumReg = Reg(UInt(outBits.W))

  val inputCounter = RegInit(0.U(1.W))

  io.bottom_out0 := topReg0
  io.bottom_out1 := topReg1
  io.right_out := leftReg
  io.sel_out := selReg
  io.psum_out := psumReg

  switch(io.state) {
    is(sIdle) {
      topReg0 := topReg0
      topReg1 := topReg1
      leftReg := leftReg
      selReg := selReg
      psumReg := psumReg
    }
    is(sClear) {
      topReg0 := 0.U
      topReg1 := 0.U
      leftReg := 0.U
      selReg := 0.U
      psumReg := 0.U
      inputCounter := 0.U
    }
    is(sCalc) {
      topReg0 := io.top_in0
      topReg1 := io.top_in1
      leftReg := leftReg
      selReg := selReg
      psumReg := Mux(selReg === 0.U, topReg0 * leftReg + io.psum_in, topReg1 * leftReg + io.psum_in)
    }
    is(sInput) {
      when(inputCounter === 1.U) {
        leftReg := io.left_in
        selReg := io.sel_in
      }
      inputCounter := inputCounter + 1.U
      topReg0 := 0.U
      topReg1 := 0.U
      psumReg := 0.U
    }
  }
}

class PrePEArray_1_2(
    bits: Int,
    point: Int,
    append: Int,
    internalBits: Int,
    width: Int,
    height: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val left_in = Input(Vec(height, UInt(4.W)))
    val top_in = Input(Vec(width, UInt(4.W)))
    val s_out = Output(Vec(height, fpType))
    val exp_sum = Output(Vec(height, fpType))
    val exp_sum_m = Output(Vec(height, fpType))
    val valid = Output(Vec(height, Bool()))

    val pes_state = Input(UInt(2.W))
    val array_state = Input(UInt(2.W))
  })

  val aIdle :: aClear :: aCalc :: Nil = Enum(3)

  val pes = (for (i <- 0 until height)
    yield for (j <- 0 until width / 2) yield Module(new PrePE_1_2(internalBits)))

  val leftFirstReg = Reg(Vec(height, UInt(4.W)))
  val cycleToggle = RegInit(0.U(1.W))

  when(cycleToggle === 0.U) {
    leftFirstReg := io.left_in
    for (i <- 0 until height) {
      pes(i)(0).io.psum_in := 0.U
      pes(i)(0).io.left_in := 0.U
      pes(i)(0).io.sel_in := 0.U
    }
    cycleToggle := 1.U
  } .otherwise {
    for (i <- 0 until height) {
      pes(i)(0).io.psum_in := 0.U
      pes(i)(0).io.left_in := Mux(leftFirstReg(i) > io.left_in(i), leftFirstReg(i), io.left_in(i))
      pes(i)(0).io.sel_in := Mux(leftFirstReg(i) > io.left_in(i), 1.U, 0.U)
    }
    cycleToggle := 0.U
  }

  for (j <- 0 until width / 2) {
    pes(0)(j).io.top_in0 := io.top_in(j * 2)
    pes(0)(j).io.top_in1 := io.top_in(j * 2 + 1)
  }

  for (r <- 0 until height) {
    for (c <- 1 until width / 2) {
      pes(r)(c).io.left_in := pes(r)(c - 1).io.right_out
      pes(r)(c).io.psum_in := pes(r)(c - 1).io.psum_out
      pes(r)(c).io.sel_in := pes(r)(c - 1).io.sel_out
    }
  }

  for (r <- 1 until height) {
    for (c <- 0 until width / 2) {
      pes(r)(c).io.top_in0 := pes(r - 1)(c).io.bottom_out0
      pes(r)(c).io.top_in1 := pes(r - 1)(c).io.bottom_out1
    }
  }

  for (r <- 0 until height; c <- 0 until width / 2) {
    pes(r)(c).io.state := io.pes_state
  }

  val exps = for (i <- 0 until height) yield Module(new ExpUnitFixPoint(bits, point, 6, 4))

  val sumRegs = Reg(Vec(height, fpType))
  val sumRegsM = Reg(Vec(height, fpType))
  val counts = Reg(Vec(height, UInt(5.W)))
  val validFlags = Reg(Vec(height, Bool()))
  val expRegs = Reg(Vec(height, fpType))

  io.s_out := expRegs
  io.exp_sum := sumRegs
  io.exp_sum_m := sumRegsM
  io.valid := validFlags

  for (i <- 0 until height) {
    if (append > 0)
      exps(i).io.in_value := Cat(
        0.U((bits - internalBits - append).W),
        pes(i)(width / 2 - 1).io.psum_out,
        0.U(append.W)
      ).asFixedPoint(point.BP)
    else
      exps(i).io.in_value := Cat(
        0.U((bits - internalBits).W),
        pes(i)(width / 2 - 1).io.psum_out
      ).asFixedPoint(point.BP)

    expRegs(i) := Mux(pes(i)(width / 2 - 1).io.psum_out === 0.U, 0.0.F(bits.W, point.BP), exps(i).io.out_exp)
  }

  switch(io.array_state) {
    is(aIdle) {
      for (i <- 0 until height) {
        sumRegs(i) := sumRegs(i)
        sumRegsM(i) := sumRegsM(i)
        expRegs(i) := expRegs(i)
        counts(i) := counts(i)
        validFlags(i) := validFlags(i)
      }
    }
    is(aClear) {
      for (i <- 0 until height) {
        sumRegs(i) := 0.F(bits.W, point.BP)
        sumRegsM(i) := 0.F(bits.W, point.BP)
        expRegs(i) := 0.F(bits.W, point.BP)
        counts(i) := 0.U
        validFlags(i) := false.B
      }
    }
    is(aCalc) {
      for (i <- 0 until height) {
        sumRegs(i) := sumRegs(i) + expRegs(i)

        when(expRegs(i) > 0.F(bits.W, point.BP)) {
          counts(i) := counts(i) + 1.U
        }

        when(counts(i) === 31.U) {
          validFlags(i) := true.B
          sumRegsM(i) := sumRegsM(i) + expRegs(i)
        } .elsewhen(counts(i) === 0.U) {
          validFlags(i) := false.B
          sumRegsM(i) := 0.F(bits.W, point.BP) + expRegs(i)
        } .otherwise {
          validFlags(i) := false.B
          sumRegsM(i) := sumRegsM(i) + expRegs(i)
        }
      }
    }
  }
}