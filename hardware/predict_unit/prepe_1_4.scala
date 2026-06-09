package predict_unit

import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint
import exp_unit.ExpUnitFixPoint

class PrePE_1_4(outBits: Int) extends Module {
  val io = IO(new Bundle {
    val left_in = Input(UInt(6.W))
    val sel_in = Input(UInt(2.W))
    val top_in0 = Input(UInt(6.W))
    val top_in1 = Input(UInt(6.W))
    val top_in2 = Input(UInt(6.W))
    val top_in3 = Input(UInt(6.W))

    val right_out = Output(UInt(6.W))
    val sel_out = Output(UInt(2.W))
    val bottom_out0 = Output(UInt(6.W))
    val bottom_out1 = Output(UInt(6.W))
    val bottom_out2 = Output(UInt(6.W))
    val bottom_out3 = Output(UInt(6.W))

    val psum_in = Input(UInt(outBits.W))
    val psum_out = Output(UInt(outBits.W))

    val state = Input(UInt(2.W))
  })

  val sIdle :: sClear :: sCalc :: sInput :: Nil = Enum(4)

  val topReg0 = Reg(UInt(6.W))
  val topReg1 = Reg(UInt(6.W))
  val topReg2 = Reg(UInt(6.W))
  val topReg3 = Reg(UInt(6.W))
  val leftReg = Reg(UInt(6.W))
  val selReg = Reg(UInt(2.W))
  val psumReg = Reg(UInt(outBits.W))

  val inputCounter = RegInit(0.U(2.W))

  io.bottom_out0 := topReg0
  io.bottom_out1 := topReg1
  io.bottom_out2 := topReg2
  io.bottom_out3 := topReg3
  io.right_out := leftReg
  io.sel_out := selReg
  io.psum_out := psumReg

  switch(io.state) {
    is(sIdle) {
      topReg0 := topReg0
      topReg1 := topReg1
      topReg2 := topReg2
      topReg3 := topReg3
      leftReg := leftReg
      selReg := selReg
      psumReg := psumReg
      inputCounter := inputCounter
    }
    is(sClear) {
      topReg0 := 0.U
      topReg1 := 0.U
      topReg2 := 0.U
      topReg3 := 0.U
      leftReg := 0.U
      selReg := 0.U
      psumReg := 0.U
      inputCounter := 0.U
    }
    is(sCalc) {
      topReg0 := io.top_in0
      topReg1 := io.top_in1
      topReg2 := io.top_in2
      topReg3 := io.top_in3
      leftReg := leftReg
      selReg := selReg
      psumReg := MuxLookup(selReg, 0.U, Seq(
        0.U -> (topReg0 * leftReg + io.psum_in),
        1.U -> (topReg1 * leftReg + io.psum_in),
        2.U -> (topReg2 * leftReg + io.psum_in),
        3.U -> (topReg3 * leftReg + io.psum_in)
      ))
    }
    is(sInput) {
      when(inputCounter === 3.U) {
        leftReg := io.left_in
        selReg := io.sel_in
      } .otherwise {
        leftReg := leftReg
        selReg := selReg
      }
      inputCounter := Mux(inputCounter === 3.U, 0.U, inputCounter + 1.U)
      topReg0 := 0.U
      topReg1 := 0.U
      topReg2 := 0.U
      topReg3 := 0.U
      psumReg := 0.U
    }
  }
}

class PrePEArray_1_4(
    bits: Int,
    point: Int,
    append: Int,
    internalBits: Int,
    width: Int,
    height: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val left_in = Input(Vec(height, UInt(6.W)))
    val top_in = Input(Vec(width, UInt(6.W)))
    val s_out = Output(Vec(height, fpType))
    val exp_sum = Output(Vec(height, fpType))
    val exp_sum_m = Output(Vec(height, fpType))
    val valid = Output(Vec(height, Bool()))

    val pes_state = Input(UInt(2.W))
    val array_state = Input(UInt(2.W))
  })

  val aIdle :: aClear :: aCalc :: Nil = Enum(3)

  require(width % 4 == 0, "width must be multiple of 4 for 1_4 PE layout")

  val pes = (for (r <- 0 until height)
    yield for (c <- 0 until width / 4) yield Module(new PrePE_1_4(internalBits)))

  val l_first = RegInit(VecInit(Seq.fill(height)(0.U(6.W))))
  val l_second = RegInit(VecInit(Seq.fill(height)(0.U(6.W))))
  val l_third = RegInit(VecInit(Seq.fill(height)(0.U(6.W))))
  val cycleCount = RegInit(0.U(2.W))

  when(cycleCount === 3.U) {
    for (i <- 0 until height) {
      pes(i)(0).io.psum_in := 0.U
      val a = io.left_in(i)
      val b = l_first(i)
      val c = l_second(i)
      val d = l_third(i)

      val max01 = Mux(a > b, a, b)
      val idx01 = Mux(a > b, 0.U(2.W), 1.U(2.W))
      val max23 = Mux(c > d, c, d)
      val idx23 = Mux(c > d, 2.U(2.W), 3.U(2.W))
      val maxAll = Mux(max01 > max23, max01, max23)
      val idxAll = Mux(max01 > max23, idx01, idx23)

      pes(i)(0).io.left_in := maxAll
      pes(i)(0).io.sel_in := idxAll
    }
    cycleCount := 0.U
    // rotate/clear shift regs after selection
    for (i <- 0 until height) {
      l_first(i) := 0.U
      l_second(i) := 0.U
      l_third(i) := 0.U
    }
  } .otherwise {
    for (i <- 0 until height) {
      pes(i)(0).io.psum_in := 0.U
      pes(i)(0).io.left_in := 0.U
      pes(i)(0).io.sel_in := 0.U

      l_third(i) := l_second(i)
      l_second(i) := l_first(i)
      l_first(i) := io.left_in(i)
    }
    cycleCount := cycleCount + 1.U
  }

  for (c <- 0 until width / 4) {
    pes(0)(c).io.top_in0 := io.top_in(c * 4)
    pes(0)(c).io.top_in1 := io.top_in(c * 4 + 1)
    pes(0)(c).io.top_in2 := io.top_in(c * 4 + 2)
    pes(0)(c).io.top_in3 := io.top_in(c * 4 + 3)
  }

  for (r <- 0 until height) {
    for (c <- 1 until width / 4) {
      pes(r)(c).io.left_in := pes(r)(c - 1).io.right_out
      pes(r)(c).io.psum_in := pes(r)(c - 1).io.psum_out
      pes(r)(c).io.sel_in := pes(r)(c - 1).io.sel_out
    }
  }

  for (r <- 1 until height) {
    for (c <- 0 until width / 4) {
      pes(r)(c).io.top_in0 := pes(r - 1)(c).io.bottom_out0
      pes(r)(c).io.top_in1 := pes(r - 1)(c).io.bottom_out1
      pes(r)(c).io.top_in2 := pes(r - 1)(c).io.bottom_out2
      pes(r)(c).io.top_in3 := pes(r - 1)(c).io.bottom_out3
    }
  }

  for (r <- 0 until height; c <- 0 until width / 4) {
    pes(r)(c).io.state := io.pes_state
  }

  val exps = for (i <- 0 until height) yield Module(new ExpUnitFixPoint(bits, point, 6, 4))

  val sumRegs = Reg(Vec(height, fpType))
  val sumRegsM = Reg(Vec(height, fpType))
  val counts = RegInit(VecInit(Seq.fill(height)(0.U(5.W))))
  val validFlags = RegInit(VecInit(Seq.fill(height)(false.B)))
  val expRegs = Reg(Vec(height, fpType))

  io.s_out := expRegs
  io.exp_sum := sumRegs
  io.exp_sum_m := sumRegsM
  io.valid := validFlags

  for (i <- 0 until height) {
    if (append > 0)
      exps(i).io.in_value := Cat(
        0.U((bits - internalBits - append).W),
        pes(i)(width / 4 - 1).io.psum_out,
        0.U(append.W)
      ).asFixedPoint(point.BP)
    else
      exps(i).io.in_value := Cat(
        0.U((bits - internalBits).W),
        pes(i)(width / 4 - 1).io.psum_out
      ).asFixedPoint(point.BP)

    expRegs(i) := Mux(pes(i)(width / 4 - 1).io.psum_out === 0.U, 0.0.F(bits.W, point.BP), exps(i).io.out_exp)
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